#!/usr/bin/env python3
"""
GRPO Training Script for 4-Stage Curriculum (Stage 2-4)

GRPO: Group Relative Policy Optimization
- Generate multiple responses per prompt
- Rank by reward
- Update policy towards better responses

Features:
- Multiple mask modes (scattered, consecutive, keep_first_last)
- Configurable reward weights
- Anti-stuffing penalty
- Token compression reward

Usage:
    /usr/bin/python3 scripts/train_grpo.py --config configs/stage2_grpo.yaml

Author: 4-Stage Curriculum Experiment
Date: 2025-12-01
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).parent))
from data_processor import CurriculumDataset, CurriculumCollator, parse_unified_format

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnswerStoppingCriteria(StoppingCriteria):
    """
    Stop generation after detecting a complete Answer (Answer: followed by value and newline).
    
    Logic:
    - Find 'Answer:' followed by a number/value
    - Stop when the answer line is complete (next newline or certain patterns after it)
    
    This prevents repetition where model outputs Answer: then starts over.
    """
    def __init__(self, tokenizer, prompt_length: int = 0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Get only the generated part (after prompt)
        if self.prompt_length > 0:
            generated = decoded[self.prompt_length:]
        else:
            generated = decoded
        
        # Stop if Question: appears (leakage)
        if 'Question:' in generated:
            return True
        
        # Check for complete Answer pattern: "Answer: <value>" followed by newline or end
        import re
        # Match Answer: followed by a value (number, possibly with $ or ,)
        answer_match = re.search(r'Answer:\s*\$?[\d,]+(?:\.\d+)?', generated)
        if answer_match:
            # Check if there's content after the answer (newline, ---, or more text)
            after_answer = generated[answer_match.end():]
            # If there's a newline after answer, stop
            if '\n' in after_answer:
                return True
            # If answer is at the very end, continue a bit to see if there's more
            # But if we see repetition patterns (-----, Step 1:, etc), stop
            if '-----' in after_answer or 'Step 1:' in after_answer:
                return True
        
        return False


class RewardCalculator:
    """
    Calculate rewards for GRPO training.
    
    Reward components:
    - correct: Answer correctness
    - format: Has Thinking/Steps/Answer structure
    - coherence: Logical flow penalty
    - token: Token compression reward
    - anti_stuffing: Penalty for step stuffing
    """
    
    def __init__(self, config: Dict):
        self.weights = config.get('reward_weights', {
            'correct': 1.0,      # 答案正确 (最重要)
            'format': 0.5,       # 格式正确
            'coherence': 0.0,    # 关闭连贯性惩罚，允许跳步压缩
            'token': 0.2,        # token压缩奖励 (正数！奖励压缩)
            'anti_stuffing': -0.5  # 反塞步惩罚
        })
        self.stuffing_threshold = config.get('anti_stuffing_threshold', 1.5)
    
    def calculate(self, response: str, gold_answer: str, 
                  original_steps: int, original_tokens: int) -> Tuple[float, Dict]:
        """
        Calculate total reward using hierarchical design.
        
        设计原则（层级递进）：
        1. 答案正确 → 基础要求，不满足直接重罚
        2. 格式完整 → 必须要求，不满足扣分
        3. 步数压缩 → 优化目标，在1和2满足后才奖励
        
        数据分析（phase1_unified_clean.jsonl）：
        - 平均步数: 4.65步
        - 2步: 13%, 3步: 30%, 4步: 22%, 5步: 13%
        - 所以 min_steps 应该设为 2（不是3！）
        
        Returns:
            (total_reward, component_dict)
        """
        rewards = {}
        
        # === 第一层：答案正确性 ===
        predicted = self._extract_answer(response)
        is_correct = self._check_answer(predicted, gold_answer)
        
        # === 第二层：格式完整性 ===
        has_step = bool(re.search(r'Step \d+:', response))
        has_answer = 'Answer:' in response or bool(re.search(r'\\boxed\{', response))
        format_ok = has_step and has_answer  # 所有Stage都必须有步骤和答案
        
        # === 第三层：步数压缩（只有前两层通过才计算）===
        response_steps = len(re.findall(r'Step \d+:', response))
        step_reduction = original_steps - response_steps  # 减少了几步
        
        # === 计算 token 数（用于奖励/惩罚）===
        response_tokens = len(response.split())
        
        # === 计算总奖励（层级递进）===
        if not is_correct:
            # 答案错误：直接重罚，其他都不重要
            total_reward = -2.0
            rewards = {
                'correct': -2.0,
                'format': 0,
                'step_reduction': 0,
                'token_reward': 0
            }
        elif not format_ok:
            # 答案对但格式不完整：扣分
            total_reward = -1.0
            rewards = {
                'correct': 1.0,  # 答案对有基础分
                'format': -2.0,  # 但格式错重扣
                'step_reduction': 0,
                'token_reward': 0
            }
        else:
            # 答案对+格式对：基础分 + 压缩奖励 + token奖励/惩罚
            base_reward = 1.0
            
            # 步数压缩奖励（每减少1步 +0.3，但不能太极端）
            if step_reduction >= 1:
                step_bonus = 0.3 * min(step_reduction, 3)  # 最多奖励减少3步
            else:
                step_bonus = 0  # 没压缩不扣分，只是没奖励
            
            # Token奖励/惩罚设计：
            # - 目标：不能超过Stage1的token，最好在80%以下
            # - 没有下限（格式+答案已约束）
            # - 超过baseline重罚，低于80%有奖励
            baseline_tokens = original_steps * 35  # Stage1每步约35 tokens
            threshold_80 = baseline_tokens * 0.8   # 80%阈值（目标线）
            
            if response_tokens > baseline_tokens:
                # 超过Stage1 baseline：重罚！不允许比Stage1更长
                excess = response_tokens - baseline_tokens
                token_reward = -0.01 * excess  # 每超1 token扣0.01（比之前重10倍）
                token_reward = max(token_reward, -1.0)  # 最多扣1.0
            elif response_tokens > threshold_80:
                # 在80%-100%之间：轻微惩罚，鼓励继续压缩
                excess = response_tokens - threshold_80
                token_reward = -0.002 * excess  # 轻微惩罚
            else:
                # 低于80%：奖励！越少越好
                reduction = threshold_80 - response_tokens
                token_reward = 0.001 * reduction  # 每少1 token加0.001
                token_reward = min(token_reward, 0.2)  # 最多加0.2
            
            total_reward = base_reward + step_bonus + token_reward
            
            rewards = {
                'correct': 1.0,
                'format': 0.2,  # 格式正确小加分
                'step_reduction': step_bonus,
                'token_reward': token_reward
            }
        
        # 记录调试信息
        rewards['_debug'] = {
            'predicted': predicted,
            'is_correct': is_correct,
            'format_ok': format_ok,
            'original_steps': original_steps,
            'response_steps': response_steps,
            'step_reduction': step_reduction,
            'response_tokens': response_tokens,
            'estimated_original_tokens': original_steps * 30
        }
        
        return total_reward, rewards
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from generated text - supports multiple formats."""
        # Try \boxed{} format first
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                return nums[0]
        
        # Try Answer: format
        match = re.search(r'Answer:\s*(.+?)$', text, re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                return nums[0]
        
        # Try = $X pattern
        eq_match = re.search(r'=\s*\$?\s*(-?\d+\.?\d*)', text)
        if eq_match:
            return eq_match.group(1)
        
        return ""
    
    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold."""
        try:
            pred_num = float(predicted.replace(',', ''))
            gold_num = float(str(gold).replace(',', ''))
            return abs(pred_num - gold_num) < 0.01
        except:
            return predicted.strip() == str(gold).strip()


def truncate_after_answer(text: str) -> str:
    """
    Truncate text after the first complete Answer.
    Handles formats: 'Answer: 18', 'Answer: $18', 'Answer: Janet makes $18...'
    """
    # Pattern: Answer: followed by optional text containing a number, then stop at newline or second Answer
    # Find first "Answer:" and the number after it
    match = re.search(r'(Answer:\s*[^\n]*?\d+[^\n]*?)(?:\n|Answer:|$)', text, re.IGNORECASE)
    if match:
        # Find where this match ends
        end_pos = match.end(1)
        return text[:end_pos].strip()
    
    # Fallback: if no Answer found, return original
    return text


class GRPOTrainer:
    """
    GRPO Trainer for Stage 2-4.
    
    Group Relative Policy Optimization:
    1. Generate multiple responses for each prompt
    2. Calculate rewards for each response
    3. Compute relative advantages within group
    4. Update policy using PPO-style objective
    """
    
    def __init__(self, config: Dict, output_dir: Path, stage: int):
        self.config = config
        self.output_dir = output_dir
        self.stage = stage
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use BF16 for full model training (FP16 has gradient issues)
        self.accelerator = Accelerator(
            mixed_precision='bf16' if config.get('use_fp16', True) else 'no'
        )
        
        # Set seed
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Load model from previous stage
        base_model_path = Path(__file__).parent.parent / config['base_model']
        logger.info(f"Loading model from: {base_model_path}")
        
        # Check if it's a LoRA adapter or full model
        adapter_config_path = base_model_path / "adapter_config.json"
        is_lora_adapter = adapter_config_path.exists()
        
        if is_lora_adapter:
            # Stage1 is LoRA format - merge first
            logger.info("Detected LoRA adapter, merging into base model...")
            import json
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            original_base_model = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-3B')
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                original_base_model,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Policy model: Load and merge Stage1, train FULL model
            base_model = AutoModelForCausalLM.from_pretrained(
                original_base_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            stage1_peft = PeftModel.from_pretrained(base_model, str(base_model_path))
            self.model = stage1_peft.merge_and_unload()
            
            # Enable training - set requires_grad=True for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Enable gradient checkpointing to save memory
            self.model.gradient_checkpointing_enable()
            logger.info("Stage1 merged - training FULL MODEL with gradient checkpointing")
            
            # Reference model: Same Stage1 merged (frozen)
            # Load separately to ensure independence
            ref_base = AutoModelForCausalLM.from_pretrained(
                original_base_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            ref_peft = PeftModel.from_pretrained(ref_base, str(base_model_path))
            self.ref_model = ref_peft.merge_and_unload()
            logger.info("Reference model loaded (Stage1 frozen copy)")
            
        else:
            # Stage1 is full model format
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_path),
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Policy model: train FULL model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            # Enable training - set requires_grad=True for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.gradient_checkpointing_enable()
            logger.info("Loaded Stage1 - training FULL MODEL with gradient checkpointing")
            
            # Reference model: frozen copy
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            logger.info("Reference model loaded (Stage1 frozen copy)")
        
        # Freeze reference model completely
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,}")
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(config)
        
        # GRPO parameters
        self.num_samples = config.get('num_samples_per_prompt', 4)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.kl_coef = config.get('kl_coef', 0.1)
        self.clip_range = config.get('clip_range', 0.2)
        
        # Training state
        self.global_step = 0
        self.best_reward = float('-inf')
    
    def train(self, resume: bool = False):
        """Run GRPO training loop."""
        # Load dataset
        logger.info("Loading training dataset...")
        data_path = Path(__file__).parent.parent / self.config['train_data_path']
        
        train_dataset = CurriculumDataset(
            str(data_path),
            self.tokenizer,
            self.config,
            stage=self.stage
        )
        
        collator = CurriculumCollator(
            self.tokenizer,
            self.config.get('max_sequence_length', 2048)
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 1),
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )
        
        # Setup optimizer
        num_epochs = self.config.get('num_epochs', 3)
        grad_accum = self.config.get('gradient_accumulation_steps', 4)
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
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Calculate start epoch and step
                steps_per_epoch = len(train_dataloader)
                start_epoch = self.global_step // steps_per_epoch
                start_step = self.global_step % steps_per_epoch
                
                # BUG FIX: Advance scheduler to correct position (same fix as Stage1 SFT)
                if self.global_step > 0:
                    logger.info(f"Advancing scheduler to step {self.global_step}...")
                    for _ in range(self.global_step):
                        scheduler.step()
                    logger.info(f"Scheduler advanced. Current LR: {scheduler.get_last_lr()[0]:.2e}")
                
                logger.info(f"Resumed from checkpoint: step={self.global_step}, epoch={start_epoch}, reward={self.best_reward:.4f}")
            else:
                logger.info("No checkpoint found, starting from scratch")
        
        # Move ref_model to GPU but DON'T prepare it (keep it independent!)
        self.ref_model = self.ref_model.to(self.accelerator.device)
        self.ref_model.eval()  # Always in eval mode
        
        # Only prepare policy model, optimizer, dataloader, scheduler
        self.model, optimizer, train_dataloader, scheduler = \
            self.accelerator.prepare(
                self.model, optimizer, train_dataloader, scheduler
            )
        
        logger.info(f"Stage {self.stage} GRPO training started")
        logger.info(f"Epochs: {num_epochs}, Samples per prompt: {self.num_samples}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_rewards = []
            epoch_kls = []
            
            for step, batch in enumerate(train_dataloader):
                # Skip steps if resuming
                if epoch == start_epoch and step < start_step:
                    continue
                
                # GRPO update
                metrics = self._grpo_step(batch, optimizer, scheduler)
                epoch_rewards.append(metrics['avg_reward'])
                epoch_kls.append(metrics.get('kl_div', 0))
                
                # 收集监控指标
                if 'correctness' in metrics:
                    if not hasattr(self, '_epoch_correctness'):
                        self._epoch_correctness = []
                        self._epoch_format_ok = []
                        self._epoch_step_reduction = []
                        self._epoch_tokens = []
                    self._epoch_correctness.append(metrics['correctness'])
                    self._epoch_format_ok.append(metrics['format_ok'])
                    self._epoch_step_reduction.append(metrics['avg_step_reduction'])
                    self._epoch_tokens.append(metrics.get('avg_response_tokens', 0))
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.get('logging_steps', 10) == 0:
                    avg_reward = sum(epoch_rewards[-10:]) / min(10, len(epoch_rewards))
                    avg_kl = sum(epoch_kls[-10:]) / min(10, len(epoch_kls))
                    
                    # 计算近10步的监控指标
                    if hasattr(self, '_epoch_correctness') and len(self._epoch_correctness) > 0:
                        recent_correct = sum(self._epoch_correctness[-10:]) / min(10, len(self._epoch_correctness))
                        recent_format = sum(self._epoch_format_ok[-10:]) / min(10, len(self._epoch_format_ok))
                        recent_step_red = sum(self._epoch_step_reduction[-10:]) / min(10, len(self._epoch_step_reduction))
                        recent_tokens = sum(self._epoch_tokens[-10:]) / min(10, len(self._epoch_tokens))
                        
                        logger.info(
                            f"Stage {self.stage} | Epoch {epoch+1}/{num_epochs} | Step {self.global_step}\n"
                            f"├─ Reward: {avg_reward:.3f} | KL: {avg_kl:.4f}\n"
                            f"├─ Correctness: {recent_correct*100:.1f}% | Format: {recent_format*100:.1f}%\n"
                            f"├─ Steps: teacher {metrics.get('original_steps', '?')} → reduction {recent_step_red:+.1f}\n"
                            f"└─ Tokens: {recent_tokens:.0f} (reward: {metrics.get('avg_token_reward', 0):+.3f})"
                        )
                    else:
                        logger.info(
                            f"Stage {self.stage} | Epoch {epoch+1}/{num_epochs} | "
                            f"Step {self.global_step} | Reward: {avg_reward:.4f} | "
                            f"KL: {avg_kl:.4f}"
                        )
                
                # Save checkpoint
                if self.global_step % self.config.get('save_steps', 50) == 0:
                    avg_reward = sum(epoch_rewards[-50:]) / min(50, len(epoch_rewards))
                    avg_kl = sum(epoch_kls[-50:]) / min(50, len(epoch_kls))
                    self._save_checkpoint(avg_reward, avg_kl)
            
            # End of epoch
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            logger.info(f"Stage {self.stage} Epoch {epoch+1} | Avg Reward: {avg_epoch_reward:.4f}")
        
        # Save final model
        self._save_final_model()
    
    def _grpo_step(self, batch: Dict, optimizer, scheduler) -> Dict:
        """
        Perform one GRPO update step.
        
        1. Generate multiple responses
        2. Calculate rewards
        3. Compute advantages
        4. PPO update
        """
        self.model.eval()
        
        # Get original sample info
        input_ids = batch['input_ids']
        
        # Generate multiple responses
        responses = []
        stopping_criteria = StoppingCriteriaList([AnswerStoppingCriteria(self.tokenizer)])
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.accelerator.unwrap_model(self.model).generate(
                    input_ids=input_ids,
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria  # Stop after Answer
                )
                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Truncate after first Answer as backup
                response_text = truncate_after_answer(response_text)
                responses.append(response_text)
        
        # Calculate rewards for each response
        # Get metadata from batch (gold_answer, num_steps, original_tokens)
        metadata_list = batch.get('metadata', [{}])
        metadata = metadata_list[0] if metadata_list else {}
        gold_answer = str(metadata.get('answer', ''))
        original_steps = metadata.get('num_steps', 5)
        # Estimate original tokens: answer + steps * avg_step_length
        original_tokens = len(gold_answer.split()) + original_steps * 25
        
        rewards = []
        reward_details = []
        for resp in responses:
            reward, details = self.reward_calculator.calculate(
                resp, 
                gold_answer=gold_answer,
                original_steps=original_steps,
                original_tokens=original_tokens
            )
            rewards.append(reward)
            reward_details.append(details)
        
        # Compute relative advantages (GRPO core)
        rewards_tensor = torch.tensor(rewards, device=self.accelerator.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        # PPO update
        self.model.train()
        
        total_loss = 0
        total_kl = 0
        for ppo_epoch in range(self.ppo_epochs):
            for i, (resp, adv) in enumerate(zip(responses, advantages)):
                # Tokenize response
                resp_encoding = self.tokenizer(
                    resp, return_tensors='pt', truncation=True, max_length=512
                ).to(self.accelerator.device)
                
                input_ids = resp_encoding['input_ids']
                attention_mask = resp_encoding['attention_mask']
                
                # Get log probs from reference model (frozen)
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    ref_logits = ref_outputs.logits
                
                # Get log probs from current policy (with grad)
                curr_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                curr_logits = curr_outputs.logits
                
                # Compute per-token log probs for the actual tokens
                # Shift logits and labels for next-token prediction
                shift_logits = curr_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_ref_logits = ref_logits[:, :-1, :].contiguous()
                
                # Get log probs for actual tokens
                curr_log_probs = F.log_softmax(shift_logits, dim=-1)
                ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
                
                # Gather log probs for actual tokens
                curr_token_log_probs = curr_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                ref_token_log_probs = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Apply attention mask (shifted)
                shift_mask = attention_mask[:, 1:].contiguous()
                curr_token_log_probs = curr_token_log_probs * shift_mask
                ref_token_log_probs = ref_token_log_probs * shift_mask
                
                # Average log prob per sequence
                seq_len = shift_mask.sum(dim=-1).clamp(min=1)
                avg_curr_log_prob = curr_token_log_probs.sum(dim=-1) / seq_len
                avg_ref_log_prob = ref_token_log_probs.sum(dim=-1) / seq_len
                
                # KL divergence: 使用绝对值确保KL惩罚总是正的
                # 原始公式 log(pi/pi_ref) 可以是负数，但我们想惩罚任何偏离
                log_ratio = avg_curr_log_prob - avg_ref_log_prob
                kl_div = log_ratio.abs().mean()  # 用绝对值！
                
                # Policy loss: maximize advantage-weighted log prob
                # Higher advantage -> want higher log prob
                policy_loss = -adv * avg_curr_log_prob.mean()
                
                # KL惩罚总是正的，防止policy偏离ref太远
                loss = policy_loss + self.kl_coef * kl_div
                
                # Normalize loss by PPO epochs and responses
                loss = loss / (self.ppo_epochs * len(responses))
                self.accelerator.backward(loss)
                total_loss += loss.item()
                total_kl += kl_div.item()
        
        # Clip gradients
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Calculate average KL over all PPO updates
        num_updates = self.ppo_epochs * len(responses)
        avg_kl = total_kl / num_updates if num_updates > 0 else 0
        
        # 统计监控信息
        correct_count = sum(1 for d in reward_details if d.get('correct', 0) > 0)
        format_ok_count = sum(1 for d in reward_details if d.get('_debug', {}).get('format_ok', False))
        step_reductions = [d.get('_debug', {}).get('step_reduction', 0) for d in reward_details]
        avg_step_reduction = sum(step_reductions) / len(step_reductions) if step_reductions else 0
        
        # Token统计
        response_tokens_list = [d.get('_debug', {}).get('response_tokens', 0) for d in reward_details]
        avg_response_tokens = sum(response_tokens_list) / len(response_tokens_list) if response_tokens_list else 0
        token_rewards = [d.get('token_reward', 0) for d in reward_details]
        avg_token_reward = sum(token_rewards) / len(token_rewards) if token_rewards else 0
        
        return {
            'avg_reward': mean_reward.item(),
            'kl_div': avg_kl,
            'loss': total_loss / num_updates if num_updates > 0 else 0,
            # 监控指标
            'correctness': correct_count / len(responses) if responses else 0,
            'format_ok': format_ok_count / len(responses) if responses else 0,
            'avg_step_reduction': avg_step_reduction,
            'original_steps': original_steps,
            'avg_response_tokens': avg_response_tokens,
            'avg_token_reward': avg_token_reward,
            'reward_details': reward_details  # 保留完整信息用于调试
        }
    
    def _save_checkpoint(self, avg_reward: float, avg_kl: float = 0.0):
        """Save checkpoint. Best is determined by reward - kl_coef * kl."""
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        
        # 计算综合得分：reward高且KL低才是真正的好
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
        logger.info(f"Checkpoint saved: step={self.global_step}, reward={avg_reward:.4f}, kl={avg_kl:.4f}, score={score:.4f}")
        
        # Run compression test every checkpoint
        self._run_compression_test()
        
        if score > self.best_reward:
            self.best_reward = score
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint! score={score:.4f}")
    
    def _run_compression_test(self):
        """
        测试模型是否学会了压缩，同时保持格式和准确率。
        
        对比Stage1和当前Stage2的输出：
        1. 格式是否正确 (Step 1/2/3... → Answer)
        2. 答案是否正确
        3. token数量是否减少（压缩）
        """
        test_problems = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "3"
            }
        ]
        
        self.model.eval()
        logger.info("=" * 60)
        logger.info("COMPRESSION TEST")
        logger.info("=" * 60)
        
        total_correct = 0
        total_format_ok = 0
        total_tokens = 0
        
        for i, problem in enumerate(test_problems):
            # 统一测试prompt：直接以Step 1:结尾，引导模型输出Step格式
            # 这是Transfer Test格式，验证模型是否学会了泛化（见§19.5）
            prompt = f"Question: {problem['question']}\n\nStep 1:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.accelerator.device)
            prompt_len = inputs['input_ids'].shape[1]
            
            stopping_criteria = StoppingCriteriaList([
                AnswerStoppingCriteria(self.tokenizer, len(prompt))
            ])
            
            with torch.no_grad():
                outputs = self.accelerator.unwrap_model(self.model).generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = truncate_after_answer(response)
            
            # Token count (generated part only)
            response_tokens = outputs.shape[1] - prompt_len
            total_tokens += response_tokens
            
            # Check format (has Step and Answer)
            has_step = bool(re.search(r'Step \d+:', response))
            has_answer = 'Answer:' in response or '\\boxed' in response
            format_ok = has_step and has_answer
            if format_ok:
                total_format_ok += 1
            
            # Check answer - support multiple formats
            # 1. Answer: $18 or Answer: 18
            answer_match = re.search(r'Answer:\s*\$?(\d+(?:\.\d+)?)', response)
            # 2. \boxed{18}
            if not answer_match:
                answer_match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', response)
            # 3. = $18 at end
            if not answer_match:
                answer_match = re.search(r'=\s*\$?(\d+(?:\.\d+)?)\s*$', response)
            predicted = answer_match.group(1) if answer_match else ""
            is_correct = predicted == problem['answer']
            if is_correct:
                total_correct += 1
            
            # Count steps
            step_count = len(re.findall(r'Step \d+:', response))
            
            logger.info(f"Test {i+1}: Format={'✓' if format_ok else '✗'} | "
                       f"Answer={'✓' if is_correct else '✗'} (pred={predicted}, gold={problem['answer']}) | "
                       f"Steps={step_count} | Tokens={response_tokens}")
            logger.info(f"  Response: {response[:200]}...")
        
        avg_tokens = total_tokens / len(test_problems)
        logger.info("-" * 60)
        logger.info(f"COMPRESSION TEST SUMMARY: "
                   f"Format={total_format_ok}/{len(test_problems)} | "
                   f"Correct={total_correct}/{len(test_problems)} | "
                   f"Avg Tokens={avg_tokens:.1f}")
        logger.info("=" * 60)
        
        self.model.train()
    
    def _save_final_model(self):
        """Save final model."""
        final_path = self.output_dir / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.unwrap_model(self.model).save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        state = {
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'config': self.config,
            'completed_at': datetime.now().isoformat()
        }
        with open(final_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        with open(self.output_dir / f"stage{self.stage}_completed.flag", 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
        
        logger.info(f"Stage {self.stage} final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--stage', type=int, default=2, help='Stage number (2, 3, or 4)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Stage {args.stage} GRPO config: {config_path}")
    
    # Setup output
    output_dir = Path(__file__).parent.parent / config.get('output_dir', f'models/stage{args.stage}')
    
    # Train
    trainer = GRPOTrainer(config, output_dir, args.stage)
    trainer.train(resume=args.resume)
    
    logger.info(f"Stage {args.stage} GRPO training completed!")


if __name__ == "__main__":
    main()
