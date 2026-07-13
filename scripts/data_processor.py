#!/usr/bin/env python3
"""
Data Processor for 4-Stage Curriculum Training

Supports multiple mask modes:
- scattered: Random mask across steps
- consecutive: Mask consecutive steps
- keep_first_last: Only keep first and last steps
- direct_qa: No CoT structure, just Question -> Answer (Stage 4)

Author: 4-Stage Curriculum Experiment
Date: 2025-12-01
Updated: 2025-12-10 - Added direct_qa mode for Stage4
"""

import re
import json
import random
import torch
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def parse_unified_format(target: str) -> Tuple[str, List[str], str]:
    """
    Parse unified format target into components.
    
    Format:
        Thinking…
        <content>
        
        Step 1: <content>
        Step 2: <content>
        ...
        
        Answer: <answer>
    
    Returns:
        (thinking, steps, answer)
    """
    # Extract Thinking section
    thinking = ""
    thinking_match = re.search(r'Thinking[…\.]+\n(.+?)(?=\nStep 1:)', target, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    # Extract Steps
    steps = []
    step_pattern = r'^Step \d+:\s*(.+?)(?=\nStep \d+:|\nAnswer:|\Z)'
    for match in re.finditer(step_pattern, target, re.MULTILINE | re.DOTALL):
        step_content = match.group(1).strip()
        if step_content:
            steps.append(step_content)
    
    # Extract Answer
    answer = ""
    answer_match = re.search(r'Answer:\s*(.+?)$', target, re.MULTILINE)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return thinking, steps, answer


def apply_mask_scattered(steps: List[str], mask_prob: float, min_mask_steps: int = 1) -> Tuple[List[str], List[int]]:
    """
    Apply scattered mask: randomly mask steps with given probability.
    
    Args:
        steps: List of step strings
        mask_prob: Probability/ratio of steps to mask
        min_mask_steps: Minimum number of steps to mask (default 1)
    
    Returns:
        (masked_steps, masked_indices)
    """
    if not steps or len(steps) < 2:
        return steps, []
    
    masked_indices = []
    # 使用max确保至少mask min_mask_steps步
    num_to_mask = max(min_mask_steps, int(len(steps) * mask_prob))
    
    # Don't mask all steps - keep at least one
    if num_to_mask >= len(steps):
        num_to_mask = len(steps) - 1
    
    masked_indices = random.sample(range(len(steps)), num_to_mask)
    
    masked_steps = []
    for i, step in enumerate(steps):
        if i in masked_indices:
            masked_steps.append("[MASKED]")
        else:
            masked_steps.append(step)
    
    return masked_steps, masked_indices


def apply_mask_consecutive(steps: List[str], mask_prob: float, min_mask_steps: int = 1) -> Tuple[List[str], List[int]]:
    """
    Apply consecutive mask: mask a continuous block of steps.
    
    Args:
        steps: List of step strings
        mask_prob: Probability/ratio of steps to mask
        min_mask_steps: Minimum number of steps to mask (default 1)
    
    Returns:
        (masked_steps, masked_indices)
    """
    if not steps or len(steps) < 2:
        return steps, []
    
    # 使用max确保至少mask min_mask_steps步
    num_to_mask = max(min_mask_steps, int(len(steps) * mask_prob))
    
    # Don't mask all steps
    if num_to_mask >= len(steps):
        num_to_mask = len(steps) - 1
    
    # Choose random start position
    max_start = len(steps) - num_to_mask
    start_idx = random.randint(0, max_start)
    
    masked_indices = list(range(start_idx, start_idx + num_to_mask))
    
    masked_steps = []
    for i, step in enumerate(steps):
        if i in masked_indices:
            masked_steps.append("[MASKED]")
        else:
            masked_steps.append(step)
    
    return masked_steps, masked_indices


def apply_mask_keep_first_last(steps: List[str]) -> Tuple[List[str], List[int]]:
    """
    Apply extreme mask: only keep first and last steps.
    
    Returns:
        (masked_steps, masked_indices)
    """
    if not steps or len(steps) <= 2:
        return steps, []
    
    masked_indices = list(range(1, len(steps) - 1))
    
    masked_steps = [steps[0]]
    for i in range(1, len(steps) - 1):
        masked_steps.append("[MASKED]")
    masked_steps.append(steps[-1])
    
    return masked_steps, masked_indices


def construct_direct_qa_input(question: str, answer: str) -> str:
    """
    Construct direct Q&A format (Stage 4).
    No CoT structure, model must reason internally.
    
    Input:  Question: xxx
            Answer:
    Target: <answer>
    """
    input_text = f"Question: {question}\nAnswer: "
    return input_text


def shuffle_steps(steps: List[str]) -> Tuple[List[str], List[int]]:
    """
    Shuffle step order and return original indices.
    
    Returns:
        (shuffled_steps, original_indices)
    """
    if not steps or len(steps) < 2:
        return steps, list(range(len(steps)))
    
    indices = list(range(len(steps)))
    random.shuffle(indices)
    
    shuffled = [steps[i] for i in indices]
    return shuffled, indices


def construct_input(question: str, thinking: str, steps: List[str], 
                   mask_mode: str, mask_prob: float, 
                   do_shuffle: bool,
                   sample_mask_probability: float = 1.0,
                   min_mask_steps: int = 1) -> Tuple[str, List[int], List[int]]:
    """
    Construct masked and optionally shuffled input.
    
    **重要修复 (2025-12-03)**：
    使用Phase1的正确格式：
    - Input中只包含Question和shuffled steps
    - Thinking在TARGET中，不在INPUT中
    - 使用中文指令，与Phase1保持一致
    
    Args:
        sample_mask_probability: Probability that this sample will be masked at all.
                                 1.0 = always mask, 0.7 = 70% samples get masked.
                                 This is for Phase1/Stage1 compatibility.
        min_mask_steps: Minimum number of steps to mask (default 1).
    
    Returns:
        (input_text, masked_indices, shuffle_order)
    """
    # Apply mask based on mode, but only with sample_mask_probability
    masked_indices = []
    if random.random() < sample_mask_probability:
        # This sample should be masked
        if mask_mode == "scattered":
            masked_steps, masked_indices = apply_mask_scattered(steps, mask_prob, min_mask_steps)
        elif mask_mode == "consecutive":
            masked_steps, masked_indices = apply_mask_consecutive(steps, mask_prob, min_mask_steps)
        elif mask_mode == "keep_first_last":
            masked_steps, masked_indices = apply_mask_keep_first_last(steps)
        elif mask_mode == "direct_qa":
            # Special mode: no steps shown at all, handled separately
            masked_steps = []
            masked_indices = list(range(len(steps)))  # All steps masked
        else:
            masked_steps = steps
    else:
        # This sample is NOT masked (pure shuffle only)
        masked_steps = steps
    
    # Apply shuffle if requested
    shuffle_order = list(range(len(masked_steps)))
    if do_shuffle and len(masked_steps) > 1:
        masked_steps, shuffle_order = shuffle_steps(masked_steps)
    
    # ========== 根据mask_mode和shuffle使用不同的prompt ==========
    # 关键设计：prompt框架保持一致，只是内容根据stage调整
    # 
    # Stage1: 有shuffle → "顺序错误且有缺失，请补充并重新排列"
    # Stage2: 无shuffle，scattered mask → "有缺失，请补充完整"
    # Stage3: keep_first_last → "只保留了首尾，请补充中间步骤"
    # Stage4: direct_qa → "没有提供任何步骤，请给出完整推理"
    
    steps_str = '\n'.join([f"[步骤] {step}" for step in masked_steps])
    
    if mask_mode == "direct_qa":
        # Stage4: 不提供任何步骤，但保持prompt框架！
        input_text = (
            f"Question: {question}\n\n"
            f"没有提供任何步骤提示，请给出完整的推理过程：\n\n"
            f"正确完整的推理过程：\n"
        )
        return input_text, masked_indices, shuffle_order
    
    if mask_mode == "keep_first_last":
        # Stage3: 只有首尾，需要补充中间
        input_text = (
            f"Question: {question}\n\n"
            f"以下步骤只保留了首尾，请补充中间缺失的步骤：\n"
            f"{steps_str}\n\n"
            f"正确完整的推理过程：\n"
        )
    elif do_shuffle:
        # Stage1格式：有shuffle
        input_text = (
            f"Question: {question}\n\n"
            f"以下步骤顺序错误且有缺失，请补充并重新排列：\n"
            f"{steps_str}\n\n"
            f"正确完整的推理过程：\n"
        )
    else:
        # Stage2格式：无shuffle，scattered/consecutive mask
        input_text = (
            f"Question: {question}\n\n"
            f"以下步骤有缺失，请补充完整：\n"
            f"{steps_str}\n\n"
            f"正确完整的推理过程：\n"
        )
    
    return input_text, masked_indices, shuffle_order


def construct_target(thinking: str, steps: List[str], answer: str) -> str:
    """
    Construct target in unified format.
    
    Target始终是完整的推理过程（Thinking + Steps + Answer），
    不管是哪个Stage，输出格式都保持一致！
    
    这样模型在Stage4也是学习输出完整推理，只是输入不提供任何步骤提示。
    """
    target_parts = ["Thinking…\n", f"{thinking}\n\n"]
    
    for i, step in enumerate(steps, 1):
        target_parts.append(f"Step {i}: {step}\n")
    
    target_parts.append(f"\nAnswer: {answer}")
    
    return "".join(target_parts)


class CurriculumDataset(Dataset):
    """
    Dataset for curriculum learning with configurable mask strategies.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict,
        stage: int = 1
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.stage = stage
        
        # Load data
        self.samples = []
        start_idx = config.get('start_index', 0)
        end_idx = config.get('end_index', None)
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if end_idx and i >= end_idx:
                    break
                sample = json.loads(line)
                self.samples.append(sample)
        
        # Mask configuration
        self.mask_prob = config.get('mask_probability', 0.15)
        self.mask_mode = config.get('mask_mode', 'scattered')
        self.do_shuffle = config.get('shuffle_steps', True)
        self.max_length = config.get('max_sequence_length', 2048)
        # sample_mask_probability: what fraction of samples get masked at all
        # Phase1/Stage1: 0.7 (70% masked, 30% pure shuffle)
        # Stage2-4: 1.0 (100% masked)
        self.sample_mask_prob = config.get('sample_mask_probability', 1.0)
        # min_mask_steps: minimum number of steps to mask
        # Stage1: 1, Stage2: 2, Stage3: 3
        self.min_mask_steps = config.get('min_mask_steps', 1)
        
        print(f"[Stage {stage}] Loaded {len(self.samples)} samples")
        print(f"[Stage {stage}] Mask mode: {self.mask_mode}, prob: {self.mask_prob}, shuffle: {self.do_shuffle}, sample_mask_prob: {self.sample_mask_prob}, min_mask_steps: {self.min_mask_steps}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        
        # Parse question and target
        question = sample['input'].replace("Question: ", "").strip()
        target = sample['target']
        
        # Parse target components
        thinking, steps, answer = parse_unified_format(target)
        
        # Apply mask and shuffle
        input_text, masked_indices, shuffle_order = construct_input(
            question, thinking, steps,
            self.mask_mode, self.mask_prob, self.do_shuffle,
            self.sample_mask_prob, self.min_mask_steps
        )
        
        # Construct target - 始终是完整的推理过程
        target_text = construct_target(thinking, steps, answer)
        
        # Tokenize
        full_text = input_text + target_text
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (mask input part)
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        input_len = len(input_encoding['input_ids'])
        
        labels = [-100] * input_len + encoding['input_ids'][input_len:]
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'metadata': {
                'idx': idx,
                'masked_indices': masked_indices,
                'shuffle_order': shuffle_order,
                'num_steps': len(steps),
                'answer': answer
            }
        }


class CurriculumCollator:
    """
    Collator for curriculum training.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        max_len = min(
            max(len(item['input_ids']) for item in batch),
            self.max_length
        )
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            
            if seq_len > max_len:
                input_ids.append(item['input_ids'][:max_len])
                attention_mask.append(item['attention_mask'][:max_len])
                labels.append(item['labels'][:max_len])
            else:
                pad_len = max_len - seq_len
                input_ids.append(item['input_ids'] + [self.pad_token_id] * pad_len)
                attention_mask.append(item['attention_mask'] + [0] * pad_len)
                labels.append(item['labels'] + [-100] * pad_len)
        
        # Collect metadata from batch items
        metadata_list = [item.get('metadata', {}) for item in batch]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'metadata': metadata_list
        }


if __name__ == "__main__":
    # Test data processor
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test config
    config = {
        'start_index': 0,
        'end_index': 10,
        'mask_probability': 0.3,
        'mask_mode': 'scattered',
        'shuffle_steps': True,
        'max_sequence_length': 1024
    }
    
    dataset = CurriculumDataset(
        "data/phase1_unified_clean.jsonl",
        tokenizer,
        config,
        stage=1
    )
    
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input length:", len(sample['input_ids']))
    print("Metadata:", sample['metadata'])
