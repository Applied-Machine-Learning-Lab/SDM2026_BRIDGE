#!/usr/bin/env python3
"""
Data processor for multi-task CoT training (Mask + Rerank).
Dynamically generates shuffled and masked inputs with three-part structured outputs.
"""

import re
import json
import random
import torch
import nltk
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer


def extract_thinking_section(cot_text: str) -> str:
    """
    Extract Thinking section from CoT text.
    
    Args:
        cot_text: Complete CoT text
    
    Returns:
        Thinking content (without markers)
    """
    # Match "Thinking..." to "...done thinking." or until "Solution:"
    match = re.search(
        r'(?i)Thinking\.{0,3}\s*(.+?)(?:\.{0,3}\s*done\s*thinking\.{0,3}|(?=\*{0,2}Solution:|\nStep\s*1:)|$)',
        cot_text,
        re.DOTALL
    )
    
    if match:
        thinking = match.group(1).strip()
        if len(thinking) > 10:  # Valid thinking content
            return thinking
    
    # Fallback 1: Extract text before "Solution:" or "Step 1:"
    match_before = re.search(
        r'^(.+?)(?:\*{0,2}Solution:|\nStep\s*1:)',
        cot_text,
        re.DOTALL
    )
    if match_before:
        content = match_before.group(1).strip()
        # Remove "Thinking..." marker if present
        content = re.sub(r'(?i)^Thinking\.{0,3}\s*', '', content)
        if len(content) > 10:
            return content
    
    # Fallback 2: Use first sentence from Solution step 1
    first_step = re.search(
        r'1\.\s*\*{0,2}[^*\n]+\*{0,2}:\s*([^.\n]+)',
        cot_text
    )
    if first_step:
        return first_step.group(1).strip()
    
    # Ultimate fallback
    return "Let me analyze this problem step by step."


def parse_steps_content(cot_text: str) -> List[str]:
    """
    Extract step contents from CoT text (without step numbers).

    Parses multiple formats:
    - "Step 1: content"
    - "1. **Title:** content"
    - GSM8K style: continuous reasoning text split by paragraphs/sentences

    Args:
        cot_text: Complete CoT text

    Returns:
        List of step contents (pure text, no numbers)
    """
    steps = []

    # Try "Step N:" format first
    step_matches = re.findall(
        r'Step\s*(\d+):\s*(.+?)(?=Step\s*\d+:|Answer:|$)',
        cot_text,
        re.DOTALL
    )

    if step_matches:
        # For Step N: format, keep all steps (don't filter by word count)
        # These are already well-structured steps from unified format
        steps = [content.strip() for num, content in step_matches]
        # Clean up LaTeX and return immediately (skip GSM8K sentence processing)
        cleaned_steps = []
        for step in steps:
            step = re.sub(r'\n\s*\n+', '\n', step)
            step = step.strip()
            step = re.sub(r'\\\[.*?\\\]', '[math]', step, flags=re.DOTALL)
            step = re.sub(r'\\\(.*?\\\)', '[math]', step, flags=re.DOTALL)
            if len(step) > 0:
                cleaned_steps.append(step)
        return cleaned_steps
    else:
        # Try numbered format "1. **Title:** content" or "1. content"
        numbered_matches = re.findall(
            r'(\d+)\.\s*(?:\*{0,2}[^*\n]+\*{0,2}:\s*)?(.+?)(?=\d+\.\s*(?:\*{0,2}[^*\n]+\*{0,2}:)?|\*{0,2}Final\s*Answer\*{0,2}:|$)',
            cot_text,
            re.DOTALL
        )

        if numbered_matches:
            steps = [content.strip() for num, content in numbered_matches]
        else:
            # GSM8K style: sentence-based splitting with semantic merging
            try:
                # Primary: NLTK sentence tokenization
                sentences = nltk.sent_tokenize(cot_text)
            except:
                # Fallback: regex-based sentence splitting
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cot_text)

            # Filter and clean sentences
            filtered_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # Skip thinking markers and final answers
                if re.match(r'(?i)^(thinking|let me|first|so|okay)', sent) and len(sent.split()) < 10:
                    continue
                if re.search(r'(?i)final answer|boxed\{', sent):
                    continue
                # Keep sentences with substantial content
                if len(sent.split()) >= 5:  # At least 5 words per sentence
                    filtered_sentences.append(sent)

            # Semantic merging: combine adjacent short sentences if semantically similar
            merged_sentences = []
            i = 0
            while i < len(filtered_sentences):
                current = filtered_sentences[i]
                current_words = len(current.split())

                # If current sentence is short (<25 words), try to merge with next
                if current_words < 25 and i + 1 < len(filtered_sentences):
                    next_sent = filtered_sentences[i + 1]
                    next_words = len(next_sent.split())
                    combined_words = current_words + next_words

                    # Only merge if combined total < 40 words (allow longer steps for GSM8K)
                    if combined_words < 40:
                        try:
                            # Load MiniLM for semantic similarity (cached)
                            if not hasattr(parse_steps_content, '_minilm_model'):
                                parse_steps_content._minilm_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

                            embeddings = parse_steps_content._minilm_model.encode([current, next_sent])
                            similarity = torch.cosine_similarity(
                                torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
                            ).item()

                            if similarity > 0.75:
                                # Merge sentences
                                merged = current + ' ' + next_sent
                                merged_sentences.append(merged)
                                i += 2  # Skip next sentence
                                continue
                        except:
                            # Fallback: keyword overlap check
                            current_tokens = set(re.findall(r'\b\w+\b', current.lower()))
                            next_tokens = set(re.findall(r'\b\w+\b', next_sent.lower()))
                            overlap = len(current_tokens & next_tokens)
                            if overlap >= 2:  # At least 2 common keywords
                                merged = current + ' ' + next_sent
                                merged_sentences.append(merged)
                                i += 2
                                continue

                # Don't merge or can't merge
                merged_sentences.append(current)
                i += 1

            steps = merged_sentences

    # Filter: keep only steps with meaningful content (>10 words for GSM8K sentences after merging)
    steps = [s for s in steps if len(s.split()) >= 10]

    # Clean up: remove excessive whitespace and markdown
    cleaned_steps = []
    for step in steps:
        # Remove excessive newlines
        step = re.sub(r'\n\s*\n+', '\n', step)
        # Remove leading/trailing whitespace
        step = step.strip()
        # Remove latex math environments for cleaner text
        step = re.sub(r'\\\[.*?\\\]', '[math]', step, flags=re.DOTALL)
        step = re.sub(r'\\\(.*?\\\)', '[math]', step, flags=re.DOTALL)

        if len(step) > 10:
            cleaned_steps.append(step)

    return cleaned_steps


def extract_answer_section(cot_text: str, metadata: Optional[Dict] = None) -> str:
    """
    Extract answer from CoT text.
    
    Args:
        cot_text: Complete CoT text
        metadata: Optional metadata containing gold_answer
    
    Returns:
        Answer text
    """
    # Try "Answer:" format
    match = re.search(
        r'(?:Answer|Final\s*Answer)\s*:\s*(.+?)$',
        cot_text,
        re.DOTALL | re.IGNORECASE
    )
    
    if match:
        answer = match.group(1).strip()
        # Clean boxed format: \(\boxed{72}\) -> 72
        answer = re.sub(r'\\[()\[\]{}]', '', answer)
        answer = re.sub(r'boxed\s*', '', answer)
        return answer.strip()
    
    # Fallback: use gold_answer from metadata
    if metadata and 'gold_answer' in metadata:
        return str(metadata['gold_answer'])
    
    return "[Answer not found]"


def construct_three_part_target(thinking: str, steps_content: List[str], answer: str) -> str:
    """
    Construct unified three-part target output.
    
    Format:
        Thinking…
        <thinking content>
        
        Step 1: <content>
        Step 2: <content>
        ...
        
        Answer: <answer>
    
    Args:
        thinking: Thinking section content
        steps_content: List of step contents (without numbers)
        answer: Final answer
    
    Returns:
        Complete three-part formatted string
    """
    # Part 1: Thinking
    thinking_part = f"Thinking…\n{thinking}\n\n"
    
    # Part 2: Steps (add numbers)
    steps_lines = []
    for i, content in enumerate(steps_content, 1):
        steps_lines.append(f"Step {i}: {content}")
    steps_part = '\n'.join(steps_lines) + "\n\n"
    
    # Part 3: Answer
    answer_part = f"Answer: {answer}"
    
    return thinking_part + steps_part + answer_part


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task CoT training.
    Loads original CoT samples from JSONL file.
    """
    
    def __init__(self, data_path: str, start_index: int = 0, end_index: int = None):
        """
        Args:
            data_path: Path to JSONL file containing CoT samples
            start_index: Starting index of samples to load (default 0)
            end_index: Ending index of samples to load (exclusive, default None = all)
        """
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_index:
                    continue
                if end_index is not None and i >= end_index:
                    break
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {data_path} (range: {start_index}-{end_index})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MultiTaskCollator:
    """
    Collator for multi-task training with dynamic mask and shuffle.
    
    For each sample:
    1. Parse three-part content (Thinking, Steps, Answer)
    2. Dynamically mask 1 step (70% probability) or no mask (30%)
    3. Shuffle all steps (100% of time)
    4. Construct input with generic [步骤] markers (no explicit step numbers)
    5. Construct target as complete three-part format
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        mask_probability: float = 0.7
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            mask_probability: Probability of masking one step (default 0.7)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Dictionary containing:
                - input_ids: [batch, seq_len]
                - labels: [batch, seq_len]
                - attention_mask: [batch, seq_len]
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for sample in batch:
            # 1. Parse three-part content
            thinking = extract_thinking_section(sample['target'])
            steps_content = parse_steps_content(sample['target'])
            answer = extract_answer_section(sample['target'], sample.get('metadata'))
            
            # Skip if parsing failed
            if len(steps_content) == 0:
                print(f"Warning: No steps found in sample, skipping")
                continue
            
            # 2. Mask+shuffle training (corrected: 70% mask, 30% pure rerank)
            # 70% mask one step, 30% no mask (pure rerank)
            masked_steps = steps_content.copy()
            
            # Mask with probability ~0.7 (not 100%)
            if random.random() < self.mask_probability:
                mask_idx = random.randint(0, len(masked_steps) - 1)
                masked_steps[mask_idx] = '<MASK>'
            
            # Always shuffle steps (100%)
            random.shuffle(masked_steps)
            
            # Construct training prompt with shuffled/masked steps
            # Remove step numbers, use generic [步骤] markers
            steps_str = '\n'.join([f"[步骤] {s}" for s in masked_steps])
            input_text = (
                f"{sample['input']}\n\n"
                f"以下步骤顺序错误且有缺失，请补充并重新排列：\n"
                f"{steps_str}\n\n"
                f"正确完整的推理过程："
            )
            
            # 3. Target is always complete CoT (regardless of input type)
            target_text = construct_three_part_target(thinking, steps_content, answer)
            
            # 4. Tokenize
            # Full text for getting complete input_ids
            full_text = input_text + " " + target_text
            full_encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            input_ids = full_encodings['input_ids'].squeeze(0)
            
            # Input-only encodings to determine where to start computing loss
            input_encodings = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            # Fix: shape is [1, seq_len], so use shape[1] to get actual length
            input_length = input_encodings['input_ids'].shape[1]
            
            # Create labels: -100 for input portion, actual tokens for target
            labels = input_ids.clone()
            labels[:input_length] = -100
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones_like(input_ids))
        
        # Pad sequences to max length in batch
        if len(batch_input_ids) == 0:
            raise ValueError("All samples in batch failed parsing")
        
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        for i in range(len(batch_input_ids)):
            pad_len = max_batch_len - len(batch_input_ids[i])
            
            padded_input_ids.append(
                torch.cat([
                    batch_input_ids[i],
                    torch.full((pad_len,), self.tokenizer.pad_token_id)
                ])
            )
            
            padded_labels.append(
                torch.cat([
                    batch_labels[i],
                    torch.full((pad_len,), -100)
                ])
            )
            
            padded_attention_mask.append(
                torch.cat([
                    batch_attention_mask[i],
                    torch.zeros(pad_len)
                ])
            )
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'labels': torch.stack(padded_labels),
            'attention_mask': torch.stack(padded_attention_mask)
        }


def test_data_processor():
    """Test function to verify data processing pipeline."""
    print("Testing data processor...")
    
    # Load one sample
    dataset = MultiTaskDataset('data/phase1_unified_clean.jsonl')
    sample = dataset[0]
    
    print("\n=== Original Sample ===")
    print(f"Input: {sample['input'][:100]}...")
    print(f"Target length: {len(sample['target'])} chars")
    
    # Test parsing
    print("\n=== Parsing ===")
    thinking = extract_thinking_section(sample['target'])
    print(f"Thinking: {thinking[:100]}...")
    
    steps = parse_steps_content(sample['target'])
    print(f"Steps: {len(steps)} steps found")
    for i, step in enumerate(steps[:3], 1):
        print(f"  Step {i}: {step[:80]}...")
    
    answer = extract_answer_section(sample['target'], sample.get('metadata'))
    print(f"Answer: {answer}")
    
    # Test three-part construction
    print("\n=== Three-Part Target ===")
    target = construct_three_part_target(thinking, steps, answer)
    print(target[:300])
    print("...")
    
    # Test dynamic mask + shuffle
    print("\n=== Dynamic Processing (3 iterations) ===")
    for iter_num in range(3):
        print(f"\nIteration {iter_num + 1}:")
        
        masked_steps = steps.copy()
        if random.random() < 0.7:
            mask_idx = random.randint(0, len(masked_steps) - 1)
            masked_steps[mask_idx] = '<MASK>'
            print(f"  Masked step {mask_idx + 1}")
        else:
            print("  No mask (30% case)")
        
        random.shuffle(masked_steps)
        print(f"  Shuffled order: {[s[:20] + '...' if len(s) > 20 else s for s in masked_steps]}")
    
    print("\n=== Test Completed ===")


if __name__ == "__main__":
    test_data_processor()
