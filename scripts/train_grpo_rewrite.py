#!/usr/bin/env python3
"""
Stage 3: GRPO for Concise Rewrite (精简重写GRPO)

任务：对Stage2无法正确回答的样本，学会看教师CoT后精简输出

与Stage2 GRPO的区别：
- Stage2: 恢复被mask的步骤
- Stage3: 看完整教师CoT后精简重写（不是恢复，是压缩！）

设计原则：
1. 使用Stage2的错误样本 + 教师CoT
2. Prompt: 给模型展示问题和教师CoT，要求精简重写
3. 奖励: 正确性 + 压缩程度（对比教师CoT长度）

Usage:
    python3 scripts/train_grpo_rewrite.py --config configs/stage3_rewrite_grpo.yaml

Date: 2025-12-20
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# GPU selection before importing torch
def select_best_gpu(mem_threshold: int = 40000) -> str:
    """Select GPU with most free memory above threshold."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )
        if result.returncode != 0:
            return "0"
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 4:
                gpu_id = int(parts[0].strip())
                mem_total = int(parts[1].strip())
                mem_used = int(parts[2].strip())
                util = int(parts[3].strip())
                mem_free = mem_total - mem_used
                gpu_info.append({
                    'id': gpu_id, 'mem_free': mem_free, 'util': util
                })
        
        candidates = [g for g in gpu_info if g['mem_free'] >= mem_threshold]
        if not candidates:
            candidates = gpu_info
        
        candidates.sort(key=lambda x: (x['util'], -x['mem_free']))
        selected = candidates[0]['id']
        
        print(f"[GPU] Status: {[(g['id'], g['mem_free'], g['util']) for g in gpu_info]}")
        print(f"[GPU] Selected GPU {selected}")
        return str(selected)
    except Exception as e:
        print(f"[GPU] Selection failed: {e}, using GPU 0")
        return "0"

GPU_ID = select_best_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import torch
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList
)
from accelerate import Accelerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnswerStoppingCriteria(StoppingCriteria):
    """Stop after detecting complete Answer."""
    def __init__(self, tokenizer, prompt_length: int = 0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if self.prompt_length > 0:
            generated = decoded[self.prompt_length:]
        else:
            generated = decoded
        
        if 'Question:' in generated:
            return True
        
        answer_match = re.search(r'Answer:\s*\$?[\d,]+(?:\.\d+)?', generated)
        if answer_match:
            after_answer = generated[answer_match.end():]
            if '\n' in after_answer or '-----' in after_answer or 'Step 1:' in after_answer:
                return True
        
        # Also check for Final Answer
        final_match = re.search(r'Final Answer:\s*\$?[\d,]+(?:\.\d+)?', generated)
        if final_match:
            after_answer = generated[final_match.end():]
            if '\n' in after_answer or len(after_answer) > 30:
                return True
        
        return False


class RewriteRewardCalculator:
    """
    Calculate rewards for Concise Rewrite GRPO.
    
    与Stage2的区别：
    - Stage2: 对比原始步数（恢复mask）
    - Stage3: 对比教师CoT长度（精简重写）
    
    奖励设计：
    - 答案正确是基础（错就重罚）
    - 比教师短才有奖励（精简的核心目标）
    """
    
    def calculate(self, response: str, gold_answer: str, 
                  teacher_tokens: int) -> Tuple[float, Dict]:
        """
        Calculate reward for concise rewrite.
        
        Args:
            response: 模型生成的精简重写
            gold_answer: 正确答案
            teacher_tokens: 教师CoT的token数（用于计算压缩比）
        """
        rewards = {}
        
        # === 第一层：答案正确性 ===
        predicted = self._extract_answer(response)
        is_correct = self._check_answer(predicted, gold_answer)
        
        # === 第二层：格式完整性 ===
        has_step = bool(re.search(r'Step \d+:', response))
        has_answer = 'Answer:' in response or 'Final Answer:' in response
        format_ok = has_step and has_answer
        
        # === 第三层：压缩程度（对比教师CoT）===
        response_tokens = len(response.split())
        compression_ratio = response_tokens / teacher_tokens if teacher_tokens > 0 else 1.0
        
        # === 计算总奖励 ===
        if not is_correct:
            total_reward = -2.0
            rewards = {
                'correct': -2.0,
                'format': 0,
                'compression': 0
            }
        elif not format_ok:
            total_reward = -1.0
            rewards = {
                'correct': 1.0,
                'format': -2.0,
                'compression': 0
            }
        else:
            # 答案对+格式对：计算压缩奖励
            base_reward = 1.0
            
            # 压缩奖励设计（越短越好，但不能太极端）
            if compression_ratio > 1.0:
                # 比教师更长：惩罚！
                compression_reward = -0.5 * min(compression_ratio - 1.0, 1.0)
            elif compression_ratio > 0.8:
                # 80%-100%: 轻微惩罚/无奖励
                compression_reward = 0
            elif compression_ratio > 0.5:
                # 50%-80%: 良好压缩，奖励
                compression_reward = 0.5 * (0.8 - compression_ratio) / 0.3
            else:
                # <50%: 优秀压缩，大奖励
                compression_reward = 0.5 + 0.3 * min((0.5 - compression_ratio) / 0.3, 1.0)
            
            total_reward = base_reward + compression_reward
            
            rewards = {
                'correct': 1.0,
                'format': 0.2,
                'compression': compression_reward
            }
        
        rewards['_debug'] = {
            'predicted': predicted,
            'is_correct': is_correct,
            'format_ok': format_ok,
            'teacher_tokens': teacher_tokens,
            'response_tokens': response_tokens,
            'compression_ratio': compression_ratio
        }
        
        return total_reward, rewards
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        # Try Final Answer first
        final_match = re.search(r'Final Answer:\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
        if final_match:
            return final_match.group(1).replace(',', '')
        
        # Try Answer:
        match = re.search(r'Answer:\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')
        
        # Try \\boxed{}
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            nums = re.findall(r'-?\d+\.?\d*', boxed_match.group(1))
            if nums:
                return nums[0]
        
        return ""
    
    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted matches gold."""
        try:
            pred_num = float(predicted.replace(',', ''))
            gold_num = float(str(gold).replace(',', ''))
            return abs(pred_num - gold_num) < 0.01
        except:
            return predicted.strip() == str(gold).strip()


class RewriteDataset(Dataset):
    """Dataset for concise rewrite task."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        
        logger.info(f"Loaded {len(self.samples)} error samples for rewrite training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        gold_answer = sample['gold_answer']
        teacher_cot = sample['teacher_cot']
        
        # 创建精简重写prompt
        prompt = self._create_rewrite_prompt(question, teacher_cot)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'metadata': {
                'question': question,
                'gold_answer': gold_answer,
                'teacher_cot': teacher_cot,
                'teacher_tokens': len(teacher_cot.split())
            }
        }
    
    def _create_rewrite_prompt(self, question: str, teacher_cot: str) -> str:
        """Create prompt for concise rewrite task."""
        return f"""You are given a math problem and its correct solution. Your task is to rewrite the solution step-by-step in a MUCH SHORTER way while keeping the same answer.

IMPORTANT: Your solution must be SHORTER than the teacher's solution. Combine steps, skip obvious calculations, and be concise.

Question: {question}

Teacher's Solution:
{teacher_cot}

Now write a SHORTER solution. Use fewer steps. End with "Final Answer: [number]"

Your Concise Solution:"""


def truncate_after_answer(text: str) -> str:
    """Truncate after first complete Answer."""
    # Try Final Answer first
    match = re.search(r'(Final Answer:\s*\$?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return text[:match.end()].strip()
    
    # Try Answer:
    match = re.search(r'(Answer:\s*\$?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return text[:match.end()].strip()
    
    return text


class RewriteGRPOTrainer:
    """GRPO Trainer for Concise Rewrite task."""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.accelerator = Accelerator(
            mixed_precision='bf16' if config.get('use_fp16', True) else 'no'
        )
        
        # Set seed
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Load model
        base_model_path = Path(__file__).parent.parent / config['base_model']
        logger.info(f"Loading model from: {base_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.gradient_checkpointing_enable()
        logger.info("Policy model loaded with gradient checkpointing")
        
        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        logger.info("Reference model loaded (frozen)")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,}")
        
        # Reward calculator
        self.reward_calculator = RewriteRewardCalculator()
        
        # GRPO parameters
        self.num_samples = config.get('num_samples_per_prompt', 2)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.kl_coef = config.get('kl_coef', 0.1)
        self.clip_range = config.get('clip_range', 0.2)
        
        # Training state
        self.global_step = 0
        self.best_reward = float('-inf')
    
    def train(self, resume: bool = False):
        """Run GRPO training loop."""
        logger.info("Loading training dataset...")
        data_path = Path(__file__).parent.parent / self.config['train_data_path']
        
        train_dataset = RewriteDataset(
            str(data_path),
            self.tokenizer,
            self.config.get('max_sequence_length', 2048)
        )
        
        def collate_fn(batch):
            # Pad sequences to max length in batch
            max_len = max(b['input_ids'].shape[0] for b in batch)
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for b in batch:
                seq_len = b['input_ids'].shape[0]
                pad_len = max_len - seq_len
                
                if pad_len > 0:
                    # Pad with pad_token_id (use 0 as fallback)
                    pad_id = self.tokenizer.pad_token_id or 0
                    padded_ids = torch.cat([
                        b['input_ids'],
                        torch.full((pad_len,), pad_id, dtype=b['input_ids'].dtype)
                    ])
                    padded_mask = torch.cat([
                        b['attention_mask'],
                        torch.zeros(pad_len, dtype=b['attention_mask'].dtype)
                    ])
                else:
                    padded_ids = b['input_ids']
                    padded_mask = b['attention_mask']
                
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
            
            input_ids = torch.stack(padded_input_ids)
            attention_mask = torch.stack(padded_attention_mask)
            metadata = [b['metadata'] for b in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'metadata': metadata
            }
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 1),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Setup optimizer
        num_epochs = self.config.get('num_epochs', 5)
        grad_accum = self.config.get('gradient_accumulation_steps', 8)
        total_steps = len(train_dataloader) * num_epochs
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # Resume from checkpoint
        start_epoch = 0
        start_step = 0
        if resume:
            checkpoint_path = self.output_dir / "checkpoint_latest.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.global_step = checkpoint.get('global_step', 0)
                self.best_reward = checkpoint.get('avg_reward', float('-inf'))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                steps_per_epoch = len(train_dataloader)
                start_epoch = self.global_step // steps_per_epoch
                start_step = self.global_step % steps_per_epoch
                
                if self.global_step > 0:
                    for _ in range(self.global_step):
                        scheduler.step()
                
                logger.info(f"Resumed from checkpoint: step={self.global_step}")
        
        # Move ref_model to GPU
        self.ref_model = self.ref_model.to(self.accelerator.device)
        
        # Prepare
        self.model, optimizer, train_dataloader, scheduler = \
            self.accelerator.prepare(
                self.model, optimizer, train_dataloader, scheduler
            )
        
        logger.info(f"Stage 3 Rewrite GRPO started")
        logger.info(f"Epochs: {num_epochs}, Samples per prompt: {self.num_samples}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_rewards = []
            epoch_kls = []
            epoch_compression = []
            epoch_correct = []
            
            for step, batch in enumerate(train_dataloader):
                if epoch == start_epoch and step < start_step:
                    continue
                
                metrics = self._grpo_step(batch, optimizer, scheduler)
                epoch_rewards.append(metrics['avg_reward'])
                epoch_kls.append(metrics.get('kl_div', 0))
                epoch_compression.append(metrics.get('avg_compression', 1.0))
                epoch_correct.append(metrics.get('correctness', 0))
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.get('logging_steps', 5) == 0:
                    avg_reward = sum(epoch_rewards[-10:]) / min(10, len(epoch_rewards))
                    avg_kl = sum(epoch_kls[-10:]) / min(10, len(epoch_kls))
                    avg_compress = sum(epoch_compression[-10:]) / min(10, len(epoch_compression))
                    avg_correct = sum(epoch_correct[-10:]) / min(10, len(epoch_correct))
                    
                    logger.info(
                        f"Stage 3 | Epoch {epoch+1}/{num_epochs} | Step {self.global_step}\n"
                        f"├─ Reward: {avg_reward:.3f} | KL: {avg_kl:.4f}\n"
                        f"├─ Correctness: {avg_correct*100:.1f}%\n"
                        f"└─ Compression: {avg_compress*100:.1f}% of teacher"
                    )
                
                # Save checkpoint
                if self.global_step % self.config.get('save_steps', 50) == 0:
                    avg_reward = sum(epoch_rewards[-50:]) / min(50, len(epoch_rewards))
                    avg_kl = sum(epoch_kls[-50:]) / min(50, len(epoch_kls))
                    self._save_checkpoint(avg_reward, avg_kl)
            
            # End of epoch
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_epoch_compress = sum(epoch_compression) / len(epoch_compression)
            logger.info(f"Stage 3 Epoch {epoch+1} | Avg Reward: {avg_epoch_reward:.4f} | Compression: {avg_epoch_compress*100:.1f}%")
        
        # Save final model
        self._save_final_model()
    
    def _grpo_step(self, batch: Dict, optimizer, scheduler) -> Dict:
        """Perform one GRPO update step. Supports batch_size > 1."""
        self.model.eval()
        
        batch_size = batch['input_ids'].shape[0]
        all_rewards = []
        all_reward_details = []
        all_responses = []
        all_advantages = []
        
        # Process each sample in the batch
        for b_idx in range(batch_size):
            input_ids = batch['input_ids'][b_idx:b_idx+1]
            attention_mask = batch['attention_mask'][b_idx:b_idx+1]
            metadata = batch['metadata'][b_idx]
            gold_answer = str(metadata.get('gold_answer', ''))
            teacher_tokens = metadata.get('teacher_tokens', 500)
            
            # Get prompt length for stopping criteria
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            prompt_length = len(prompt_text)
            
            # Generate multiple responses for this sample
            responses = []
            stopping_criteria = StoppingCriteriaList([
                AnswerStoppingCriteria(self.tokenizer, prompt_length)
            ])
            
            with torch.no_grad():
                for _ in range(self.num_samples):
                    outputs = self.accelerator.unwrap_model(self.model).generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        stopping_criteria=stopping_criteria
                    )
                    
                    generated_ids = outputs[0][input_ids.shape[1]:]
                    response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    response_text = truncate_after_answer(response_text)
                    responses.append(response_text)
            
            # Calculate rewards for this sample's responses
            rewards = []
            reward_details = []
            for resp in responses:
                reward, details = self.reward_calculator.calculate(
                    resp,
                    gold_answer=gold_answer,
                    teacher_tokens=teacher_tokens
                )
                rewards.append(reward)
                reward_details.append(details)
            
            # Compute advantages within this sample's group
            rewards_tensor = torch.tensor(rewards, device=self.accelerator.device)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_reward) / std_reward
            
            all_rewards.extend(rewards)
            all_reward_details.extend(reward_details)
            all_responses.extend(responses)
            all_advantages.extend(advantages.tolist())
        
        # PPO update with all samples
        self.model.train()
        
        total_loss = 0
        total_kl = 0
        num_updates = 0
        
        for ppo_epoch in range(self.ppo_epochs):
            for i, (resp, adv) in enumerate(zip(all_responses, all_advantages)):
                resp_encoding = self.tokenizer(
                    resp, return_tensors='pt', truncation=True, max_length=512
                ).to(self.accelerator.device)
                
                resp_ids = resp_encoding['input_ids']
                resp_mask = resp_encoding['attention_mask']
                
                # Reference model log probs
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=resp_ids,
                        attention_mask=resp_mask
                    )
                    ref_logits = ref_outputs.logits
                
                # Current policy log probs
                curr_outputs = self.model(
                    input_ids=resp_ids,
                    attention_mask=resp_mask
                )
                curr_logits = curr_outputs.logits
                
                # Compute per-token log probs
                shift_logits = curr_logits[:, :-1, :].contiguous()
                shift_labels = resp_ids[:, 1:].contiguous()
                shift_ref_logits = ref_logits[:, :-1, :].contiguous()
                
                curr_log_probs = F.log_softmax(shift_logits, dim=-1)
                ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
                
                curr_token_log_probs = curr_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                ref_token_log_probs = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                shift_mask = resp_mask[:, 1:].contiguous()
                curr_token_log_probs = curr_token_log_probs * shift_mask
                ref_token_log_probs = ref_token_log_probs * shift_mask
                
                seq_len = shift_mask.sum(dim=-1).clamp(min=1)
                avg_curr_log_prob = curr_token_log_probs.sum(dim=-1) / seq_len
                avg_ref_log_prob = ref_token_log_probs.sum(dim=-1) / seq_len
                
                log_ratio = avg_curr_log_prob - avg_ref_log_prob
                kl_div = log_ratio.abs().mean()
                
                adv_tensor = torch.tensor(adv, device=self.accelerator.device)
                policy_loss = -adv_tensor * avg_curr_log_prob.mean()
                loss = policy_loss + self.kl_coef * kl_div
                loss = loss / (self.ppo_epochs * len(all_responses))
                
                self.accelerator.backward(loss)
                total_loss += loss.item()
                total_kl += kl_div.item()
                num_updates += 1
        
        # Clip gradients
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('max_grad_norm', 1.0)
            )
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Compute metrics
        avg_kl = total_kl / num_updates if num_updates > 0 else 0
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        
        correct_count = sum(1 for d in all_reward_details if d.get('correct', 0) > 0)
        compression_ratios = [d.get('_debug', {}).get('compression_ratio', 1.0) for d in all_reward_details]
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
        
        return {
            'avg_reward': avg_reward,
            'kl_div': avg_kl,
            'correctness': correct_count / len(all_responses) if all_responses else 0,
            'avg_compression': avg_compression
        }
    
    def _save_checkpoint(self, avg_reward: float, avg_kl: float = 0.0):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        score = avg_reward - self.kl_coef * avg_kl
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'avg_reward': avg_reward,
            'avg_kl': avg_kl,
            'score': score,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: step={self.global_step}, reward={avg_reward:.4f}, kl={avg_kl:.4f}")
        
        # Save best
        if score > self.best_reward:
            self.best_reward = score
            best_path = self.output_dir / f"checkpoint_best_step{self.global_step}.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint! Score: {score:.4f}")
    
    def _save_final_model(self):
        """Save final model."""
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(__file__).parent.parent / config.get('output_dir', 'models/stage3_rewrite')
    
    trainer = RewriteGRPOTrainer(config, output_dir)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
