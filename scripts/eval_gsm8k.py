#!/usr/bin/env python3
"""
GSM8K evaluation script for BRIDGE checkpoints.

Greedy decoding, max_new_tokens=512, with AnswerStoppingCriteria (stops once a
final answer is emitted). Reports accuracy and average generated token count on
the GSM8K test set. Model path / output dir / GPU are given on the command line:

    python scripts/eval_gsm8k.py --model_path models/stage3_rewrite_v2/final_model
"""

import os
import sys
import json
import torch
import re
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# GPU选择 - 只选择一个GPU
def select_best_gpu():
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 3:
                idx, free_mem, util = int(parts[0]), int(parts[1]), int(parts[2])
                if free_mem > 20000:
                    gpus.append((idx, free_mem, util))
        if not gpus:
            return 0
        gpus.sort(key=lambda x: (x[2], -x[1]))
        print(f"Selected GPU {gpus[0][0]}: {gpus[0][1]}MB free, {gpus[0][2]}% util")
        return gpus[0][0]
    except Exception as e:
        print(f"GPU selection error: {e}")
        return 0

_parser = argparse.ArgumentParser(description="GSM8K evaluation for BRIDGE models")
_parser.add_argument('--model_path', required=True, help='HF checkpoint dir to evaluate')
_parser.add_argument('--num_samples', type=int, default=1319, help='number of GSM8K test samples (max 1319)')
_parser.add_argument('--output_dir', default='eval_results/bridge_eval', help='where to write results/checkpoint')
_parser.add_argument('--gpu', type=int, default=None, help='CUDA device id; default: auto-select a free GPU')
_args = _parser.parse_args()

GPU_ID = _args.gpu if _args.gpu is not None else select_best_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset

MODEL_PATH = _args.model_path
OUTPUT_DIR = Path(_args.output_dir)
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
NUM_SAMPLES = _args.num_samples


class AnswerStoppingCriteria(StoppingCriteria):
    """停止条件：检测到Answer后停止"""
    def __init__(self, tokenizer, prompt_length=0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if self.prompt_length > 0:
            generated = decoded[self.prompt_length:]
        else:
            generated = decoded
        
        # 停止条件1: 检测到新问题
        if 'Question:' in generated:
            return True
        
        # 停止条件2: 检测到Answer后换行
        answer_match = re.search(r'(Final Answer|Answer):\s*\$?[\d,]+(?:\.\d+)?', generated, re.IGNORECASE)
        if answer_match:
            after = generated[answer_match.end():]
            if '\n' in after or len(after) > 20:
                return True
        
        return False


def extract_answer(text):
    """提取答案"""
    match = re.search(r'Final Answer:\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    match = re.search(r'Answer:\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return numbers[-1] if numbers else ""


def extract_gold_answer(answer_text):
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        return match.group(1)
    return ""


def create_prompt(question):
    """创建评估prompt - 必须包含Final Answer要求"""
    return f"""Solve this math problem step by step, showing your complete reasoning.
Question: {question}
You must end your response with "Final Answer: [number]"
Reasoning:"""


def check_output_quality(response):
    """检查输出质量"""
    issues = []
    
    if len(re.findall(r'Step 1:', response)) > 1:
        issues.append("repeated_step1")
    
    weird_chars = len(re.findall(r'[^\x00-\x7F]', response))
    if weird_chars > 10:
        issues.append("weird_chars")
    
    if '...' * 3 in response or '---' * 5 in response:
        issues.append("loop_pattern")
    
    has_step = bool(re.search(r'Step \d+:', response))
    has_answer = bool(re.search(r'(Final Answer|Answer):', response, re.IGNORECASE))
    
    if not has_step:
        issues.append("no_step")
    if not has_answer:
        issues.append("no_answer")
    
    return issues


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Stage3 GRPO v2 模型评估 (验证版)")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Samples: {NUM_SAMPLES}")
    
    # 加载模型
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")
    
    # 加载GSM8K测试集
    print("\nLoading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(dataset)[:NUM_SAMPLES]
    print(f"Evaluating on {len(samples)} samples")
    
    # 检查checkpoint
    results = []
    completed_indices = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        results = checkpoint.get('results', [])
        completed_indices = set(r['idx'] for r in results)
        print(f"Resuming from checkpoint: {len(completed_indices)} completed")
    
    # 评估
    print("\n开始评估...")
    quality_issues = {}
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        if idx in completed_indices:
            continue
        
        question = sample['question']
        gold = extract_gold_answer(sample['answer'])
        
        prompt = create_prompt(question)
        prompt_length = len(prompt)
        
        inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        stopping_criteria = StoppingCriteriaList([
            AnswerStoppingCriteria(tokenizer, prompt_length)
        ])
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = len(generated_ids)
        except Exception as e:
            response = f"ERROR: {e}"
            token_count = 0
        
        predicted = extract_answer(response)
        is_correct = (predicted == gold)
        
        issues = check_output_quality(response)
        for issue in issues:
            quality_issues[issue] = quality_issues.get(issue, 0) + 1
        
        results.append({
            'idx': idx,
            'question': question[:100],
            'gold': gold,
            'predicted': predicted,
            'is_correct': is_correct,
            'token_count': token_count,
            'response': response[:500],
            'issues': issues
        })
        
        # 每20个保存checkpoint
        if (idx + 1) % 20 == 0:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({'results': results}, f)
    
    # 最终保存
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'results': results}, f)
    
    # 统计
    correct_count = sum(1 for r in results if r['is_correct'])
    total_tokens = sum(r['token_count'] for r in results)
    accuracy = correct_count / len(results) * 100
    avg_tokens = total_tokens / len(results)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Samples: {len(results)}")
    print(f"Correct: {correct_count} ({accuracy:.2f}%)")
    print(f"Average tokens: {avg_tokens:.1f}")
    
    print(f"\n=== 输出质量检查 ===")
    for issue, count in sorted(quality_issues.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count}")
    
    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"grpo_eval_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'model': MODEL_PATH,
                'samples': len(results),
                'correct': correct_count,
                'accuracy': accuracy,
                'avg_tokens': avg_tokens,
                'quality_issues': quality_issues
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # 显示样本
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (first 3)")
    print("=" * 70)
    for i, r in enumerate(results[:3]):
        print(f"\n--- Sample {i+1} {'✓' if r['is_correct'] else '✗'} ---")
        print(f"Gold: {r['gold']}, Predicted: {r['predicted']}")
        print(f"Tokens: {r['token_count']}, Issues: {r['issues']}")
        print(f"Response: {r['response'][:300]}...")


if __name__ == '__main__':
    main()
