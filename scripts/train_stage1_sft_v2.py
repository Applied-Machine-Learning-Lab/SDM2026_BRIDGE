#!/usr/bin/env python3
"""
Multi-Task CoT Training Script (Mask + Rerank)

Target: Train Qwen2.5-3B to learn three-part CoT output (Thinking + Step + Answer)
through dynamically masked and shuffled inputs.

Features:
- Dynamic mask and shuffle data augmentation
- Three-part format monitoring
- LoRA efficient fine-tuning
- Checkpoint/Resume functionality
- Guardian compatible

Usage:
    python3 scripts/train_multi_task.py \
        --config configs/multi_task_config.yaml \
        --resume
"""

# !!!CRITICAL!!! Must set environment variables before importing torch
import os
import sys
import subprocess
from typing import Optional

GPU_MEM_FREE_THRESHOLD_DEFAULT = 40000
_gpu_mem_threshold = GPU_MEM_FREE_THRESHOLD_DEFAULT

def select_free_gpu(mem_threshold: Optional[int] = None) -> str:
    """Select GPU with sufficient free memory, then minimal utilization"""
    mem_threshold = mem_threshold if mem_threshold is not None else GPU_MEM_FREE_THRESHOLD_DEFAULT
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print(f"[WARN] nvidia-smi failed, using GPU 0 by default", flush=True)
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
                    'id': gpu_id,
                    'mem_total': mem_total,
                    'mem_used': mem_used,
                    'mem_free': mem_free,
                    'util': util
                })
        
        if not gpu_info:
            print(f"[WARN] No GPU info found, using GPU 0 by default", flush=True)
            return "0"
        
        # Filter by free memory threshold first
        candidates = [info for info in gpu_info if info['mem_free'] >= mem_threshold]
        if not candidates:
            print(
                f"[WARN] No GPU satisfies free memory threshold {mem_threshold} MB, considering all GPUs",
                flush=True
            )
            candidates = gpu_info
        
        # Select GPU with lowest utilization, tie-breaker highest free memory
        candidates.sort(key=lambda info: (info['util'], -info['mem_free']))
        selected = candidates[0]
        selected_id = str(selected['id'])
        
        readable = [(info['id'], info['mem_free'], info['mem_used'], info['util']) for info in gpu_info]
        print(f"[INIT] GPU status (id, freeMB, usedMB, util%): {readable}", flush=True)
        print(
            f"[INIT] Auto-selected GPU {selected_id} "
            f"(Free mem: {selected['mem_free']} MB, Util: {selected['util']}%, "
            f"Mem threshold: {mem_threshold} MB)",
            flush=True
        )
        return selected_id
        
    except Exception as e:
        print(f"[WARN] Failed to select GPU: {e}, using GPU 0 by default", flush=True)
        return "0"

# Dynamic GPU selection (unless explicitly specified in config)
_cuda_device = None
for i, arg in enumerate(sys.argv):
    if arg == '--config' and i + 1 < len(sys.argv):
        import yaml
        with open(sys.argv[i + 1], 'r') as f:
            _temp_config = yaml.safe_load(f)
            _cuda_device = _temp_config.get('cuda_visible_devices', None)
            _gpu_mem_threshold = _temp_config.get('gpu_mem_free_threshold', GPU_MEM_FREE_THRESHOLD_DEFAULT)
        break

if _cuda_device is None or _cuda_device == "auto":
    _cuda_device = select_free_gpu(_gpu_mem_threshold)
else:
    _cuda_device = str(_cuda_device)
    print(f"[INIT] Config specified GPU: {_cuda_device}", flush=True)

# Set environment variables (before importing torch)
os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_device
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"[INIT] Set CUDA_VISIBLE_DEVICES={_cuda_device}", flush=True)
print(f"[INIT] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", flush=True)

# Now import torch-related modules
import json
import torch
import random
import logging
import argparse
import re
from typing import Dict, List
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

# Import custom modules - 使用Phase1的data_processor
from data_processor_phase1 import MultiTaskDataset, MultiTaskCollator


class AnswerStoppingCriteria(StoppingCriteria):
    """
    Stop generation after detecting a complete Answer (Answer: followed by value and newline).
    
    This ensures clean output: Thinking -> Steps -> Answer (done)
    Prevents repetition where model outputs Answer: then starts over with '-----' separator.
    """
    def __init__(self, tokenizer, prompt_length: int = 0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids, scores, **kwargs):
        # Decode generated text
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Get only the generated part (after prompt)
        if self.prompt_length > 0:
            generated = decoded[self.prompt_length:]
        else:
            generated = decoded
        
        # Stop if Question: appears (leakage - starting another problem)
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
            # If we see repetition patterns (-----, Step 1:, etc), stop
            if '-----' in after_answer or 'Step 1:' in after_answer:
                return True
        
        return False


def truncate_after_answer(text: str) -> str:
    """
    Truncate text after the first complete Answer.
    Handles formats: 'Answer: 18', 'Answer: $18', 'Answer: The result is 18.'
    
    Returns clean output ending at the first answer.
    """
    # Find first Answer: and keep everything up to end of that line or next Answer:/Question:
    match = re.search(r'(Answer:\s*[^\n]*?\d+[^\n]*?)(?:\n|Answer:|Question:|$)', text, re.IGNORECASE)
    if match:
        end_pos = match.end(1)
        return text[:end_pos].strip()
    
    # Fallback: if Answer: exists but no number, keep until newline
    match2 = re.search(r'(Answer:[^\n]+)', text, re.IGNORECASE)
    if match2:
        return text[:match2.end()].strip()
    
    # No Answer found, return original
    return text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """
    Multi-task CoT trainer.
    Integrates Mask + Rerank in a unified framework.
    """
    
    def __init__(self, config: Dict, resume: bool = False, checkpoint_dir: Optional[str] = None):
        """
        Args:
            config: Training configuration dictionary
            resume: Whether to resume from checkpoint
            checkpoint_dir: Directory containing checkpoints
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.get('output_dir', 'models/multi_task_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Three-part format monitoring
        self.format_stats = {
            'thinking_count': 0,
            'step_count': 0,
            'answer_count': 0,
            'total_checks': 0
        }
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 16),
            mixed_precision='fp16' if config.get('use_fp16', True) else 'no',
            log_with='tensorboard' if config.get('use_tensorboard', True) else None,
            project_dir=str(Path(config.get('logging_dir', 'logs/multi_task_training')))
        )
        
        logger.info(f"Accelerator initialized: device={self.accelerator.device}, num_processes={self.accelerator.num_processes}")
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
        # Resume from checkpoint if requested
        if resume:
            self._load_checkpoint()
        
        logger.info(f"Trainer initialized - Resume={resume}, GlobalStep={self.global_step}, Epoch={self.epoch}")
    
    def _load_model(self):
        """Load tokenizer and model with LoRA"""
        logger.info("Loading tokenizer and model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            cache_dir=self.config.get('cache_dir', 'hf_cache'),
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Ensure pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            cache_dir=self.config.get('cache_dir', 'hf_cache'),
            torch_dtype=torch.float16 if self.config.get('use_fp16', True) else torch.float32,
            device_map=None,  # Let accelerate handle device assignment
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
            local_files_only=True
        )
        
        # Enable gradient checkpointing to reduce memory
        if self.config.get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # LoRA configuration (MODIFIED: r=16, alpha=32, no rslora)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get('lora_r', 16),  # CHANGED from 32 to 16
            lora_alpha=self.config.get('lora_alpha', 32),  # CHANGED from 16 to 32
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return tokenizer, model
    
    def _load_checkpoint(self):
        """Load checkpoint to resume training"""
        checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        
        # Try loading checkpoint_latest.pt first
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load training state
                self.global_step = checkpoint.get('global_step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                
                logger.info(f"Resumed from checkpoint_latest.pt: step={self.global_step}, epoch={self.epoch}, best_loss={self.best_loss:.4f}")
                return
            
            except Exception as e:
                logger.error(f"Failed to load checkpoint_latest.pt: {e}")
                logger.warning("checkpoint_latest.pt is corrupted, trying checkpoint_best.pt...")
        
        # Fallback to checkpoint_best.pt if latest is missing or corrupted
        if best_checkpoint_path.exists():
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load training state
                self.global_step = checkpoint.get('global_step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                
                logger.info(f"Resumed from checkpoint_best.pt: step={self.global_step}, epoch={self.epoch}, best_loss={self.best_loss:.4f}")
                logger.warning("Using checkpoint_best.pt as checkpoint_latest.pt was corrupted")
                return
            
            except Exception as e:
                logger.error(f"Failed to load checkpoint_best.pt: {e}")
        
        logger.warning("No valid checkpoint found, starting training from scratch")
    
    def save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """Save checkpoint with atomic write (write to temp file first, then move)"""
        import shutil
        
        checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        temp_checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt.tmp"
        
        # Get unwrapped model (in case using accelerator or DDP)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        checkpoint = {
            'model_state_dict': unwrapped_model.state_dict(),
            'global_step': step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Atomic save: write to temp file first, then move (prevents corruption on kill)
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use absolute paths to avoid issues with working directory
            temp_checkpoint_path_abs = temp_checkpoint_path.resolve()
            checkpoint_path_abs = checkpoint_path.resolve()
            
            torch.save(checkpoint, str(temp_checkpoint_path_abs))
            # Move is atomic on most filesystems
            shutil.move(str(temp_checkpoint_path_abs), str(checkpoint_path_abs))
            logger.info(f"Checkpoint saved: step={step}, loss={loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_checkpoint_path.exists():
                temp_checkpoint_path.unlink()
            raise
        
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            temp_best_path = self.checkpoint_dir / "checkpoint_best.pt.tmp"
            
            try:
                # Use absolute paths
                temp_best_path_abs = temp_best_path.resolve()
                best_path_abs = best_path.resolve()
                
                torch.save(checkpoint, str(temp_best_path_abs))
                shutil.move(str(temp_best_path_abs), str(best_path_abs))
                logger.info(f"Best model saved with loss={loss:.4f}")
            except Exception as e:
                logger.error(f"Failed to save best checkpoint: {e}")
                if temp_best_path.exists():
                    temp_best_path.unlink()
    
    def check_three_part_format(self, text: str) -> Dict[str, bool]:
        """Check if generated text contains all three parts"""
        has_thinking = bool(re.search(r'(?i)Thinking\.{0,3}', text))
        has_step = bool(re.search(r'Step\s*\d+:', text))
        has_answer = bool(re.search(r'(?i)Answer\s*:', text))
        
        return {
            'thinking': has_thinking,
            'step': has_step,
            'answer': has_answer,
            'all_three': has_thinking and has_step and has_answer
        }
    
    def train(self):
        """Main training loop"""
        # Load dataset with data range support for curriculum learning
        logger.info("Loading training dataset...")
        train_dataset = MultiTaskDataset(
            data_path=str(Path(self.config['train_data_path'])),
            start_index=self.config.get('start_index', 0),
            end_index=self.config.get('end_index', None)
        )
        
        # Create collator
        collator = MultiTaskCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_sequence_length', 4096),
            mask_probability=self.config.get('mask_probability', 0.7)
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 1),
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=False
        )
        
        # Setup optimizer and scheduler
        num_epochs = self.config.get('num_epochs', 5)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 16)
        # Total training steps = total batches / gradient_accumulation_steps
        total_steps = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 80),
            num_training_steps=total_steps
        )
        
        # If resuming, advance scheduler to correct position
        if self.global_step > 0:
            logger.info(f"Advancing scheduler to step {self.global_step}...")
            for _ in range(self.global_step):
                scheduler.step()
            logger.info(f"Scheduler advanced. Current LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Prepare with accelerator
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        logger.info(f"Training started: {num_epochs} epochs, {len(train_dataloader)} batches/epoch")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Mask probability: {self.config.get('mask_probability', 0.7)}")
        
        # Training loop
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 16)
        # Calculate steps per epoch for epoch tracking
        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        
        for epoch in range(num_epochs):
            # Calculate expected epoch from global_step
            expected_epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
            if epoch < expected_epoch:
                continue  # Skip completed epochs
            
            self.epoch = epoch
            self.model.train()
            
            total_loss = 0
            accumulation_counter = 0  # Track batches for manual gradient accumulation
            
            for step, batch in enumerate(train_dataloader):
                # Skip processed batches based on global_step
                # Each gradient_accumulation_steps batches = 1 training step
                # So: batch_index corresponding to global_step = global_step * gradient_accumulation_steps
                current_batch_index = epoch * len(train_dataloader) + step
                target_batch_index = self.global_step * gradient_accumulation_steps
                
                if current_batch_index < target_batch_index:
                    continue
                
                # Manual gradient accumulation (don't use accelerator.accumulate for resume compatibility)
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # CE loss (main training objective - pure CE only)
                # Scale loss by gradient_accumulation_steps
                total_loss_step = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(total_loss_step)
                
                accumulation_counter += 1
                total_loss += outputs.loss.item()  # Track unscaled loss for logging
                
                # Only update weights every gradient_accumulation_steps batches
                if accumulation_counter >= gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.get('max_grad_norm', 1.0) > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    self.global_step += 1
                    accumulation_counter = 0
                    
                    # Update epoch based on current step
                    self.epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                    
                    # Logging
                    logging_steps = self.config.get('logging_steps', 10)
                    if self.global_step % logging_steps == 0:
                        avg_loss = total_loss / gradient_accumulation_steps / logging_steps
                        
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | Step {self.global_step}: "
                            f"Loss={avg_loss:.4f}, "
                            f"LR={scheduler.get_last_lr()[0]:.2e}"
                        )
                        
                        total_loss = 0
                    
                    # Three-part format monitoring (every 100 steps)
                    if self.global_step % 100 == 0:
                        self._monitor_three_part_format()
                    
                    # Save checkpoint
                    if self.global_step % self.config.get('save_steps', 100) == 0:
                        current_loss = outputs.loss.item()
                        is_best = current_loss < self.best_loss
                        if is_best:
                            self.best_loss = current_loss
                        
                        self.save_checkpoint(self.global_step, current_loss, is_best)
        
        # Save final model - MERGE LoRA into base model for Stage2 GRPO!
        # Stage2需要从merged的完整模型开始，这样加新LoRA时KL才会大
        logger.info("Training completed! Saving merged final model...")
        final_model_path = self.checkpoint_dir / "final_model"
        
        # Merge LoRA weights into base model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        merged_model = unwrapped_model.merge_and_unload()
        logger.info("LoRA merged into base model")
        
        # Save merged full model (not LoRA adapter!)
        merged_model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'format_stats': self.format_stats,
            'config': self.config
        }
        with open(final_model_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Create completion flag for auto-resubmit detection
        completion_flag = self.checkpoint_dir / "training_completed.flag"
        with open(completion_flag, 'w') as f:
            f.write(f"Training completed at: {datetime.now().isoformat()}\n")
            f.write(f"Total steps: {self.global_step}\n")
            f.write(f"Total epochs: {self.epoch + 1}\n")
        
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Completion flag created: {completion_flag}")
        
        # Print final format stats
        if self.format_stats['total_checks'] > 0:
            thinking_rate = self.format_stats['thinking_count'] / self.format_stats['total_checks']
            step_rate = self.format_stats['step_count'] / self.format_stats['total_checks']
            answer_rate = self.format_stats['answer_count'] / self.format_stats['total_checks']
            
            logger.info(f"\n=== Final Three-Part Format Stats ===")
            logger.info(f"Total checks: {self.format_stats['total_checks']}")
            logger.info(f"Thinking rate: {thinking_rate:.2%}")
            logger.info(f"Step rate: {step_rate:.2%}")
            logger.info(f"Answer rate: {answer_rate:.2%}")
    
    def _monitor_three_part_format(self):
        """Monitor if model outputs contain three-part format
        
        Two tests are performed:
        1. Task Learning Test: Using training-style prompt (mask+shuffle)
        2. Transfer Test: Using a NEW question with format guidance (no mask/shuffle)
        
        This helps monitor both task learning AND transfer capability.
        """
        self.model.eval()
        
        # Test 1: Task Learning - training-style prompt (mask+shuffle)
        # Use a question from training data
        task_prompt = """Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

以下步骤顺序错误且有缺失，请补充并重新排列：
[步骤] Total Pages Read So Far: [math]
[步骤] <MASK>
[步骤] Pages Read Yesterday: 12

正确完整的推理过程："""
        
        # Test 2: Transfer - a DIFFERENT question with format guidance
        # This tests if model can generalize to new problems
        # Answer should be 35
        transfer_prompt = """Question: A baker has 200 cookies. She sells 40% of them in the morning and 1/3 of the remaining in the afternoon. How many cookies does she have left?

请按以下格式输出推理过程：Thinking…思考过程，Step 1/2/3...推理步骤，Answer: 答案

Thinking…"""
        
        try:
            # Run Task Learning Test
            task_result = self._run_format_test(task_prompt, "TaskLearning")
            
            # Run Transfer Test
            transfer_result = self._run_format_test(transfer_prompt, "Transfer")
            
            # Update stats (using transfer result as main metric)
            self.format_stats['total_checks'] += 1
            if transfer_result['thinking']:
                self.format_stats['thinking_count'] += 1
            if transfer_result['step']:
                self.format_stats['step_count'] += 1
            if transfer_result['answer']:
                self.format_stats['answer_count'] += 1
            
            # Log both results
            logger.info(
                f"[ThreePartCheck] Step {self.global_step}: "
                f"TaskLearning(Step={task_result['step']}) | "
                f"Transfer(Thinking={transfer_result['thinking']}, Step={transfer_result['step']}, Answer={transfer_result['answer']})"
            )
        
        except Exception as e:
            logger.warning(f"Three-part format check failed: {e}")
        
        finally:
            self.model.train()
    
    def _run_format_test(self, prompt: str, test_name: str) -> Dict[str, bool]:
        """Run a single format test and return results.
        
        Uses StoppingCriteria to stop generation after:
        1. Second Answer: appears (repetition)
        2. Question: appears (leakage)
        
        Also applies truncate_after_answer() for clean output.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.accelerator.device)
        
        # Create stopping criteria to prevent repetition/leakage
        prompt_length = len(prompt)
        stopping_criteria = StoppingCriteriaList([
            AnswerStoppingCriteria(self.tokenizer, prompt_length)
        ])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(prompt):].strip()
        
        # Apply truncation to ensure clean output
        generated_part = truncate_after_answer(generated_part)
        
        return self.check_three_part_format(generated_part)


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task CoT Qwen2.5-3B (Mask+Rerank, CE-only)')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint_dir', default=None, help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to: {seed}")
    
    # Create trainer
    trainer = MultiTaskTrainer(
        config=config,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    trainer.train()
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()

