#!/usr/bin/env python3
"""Read-only integrity check for the BRIDGE open-source release."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PROBLEMS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        PROBLEMS.append(message)


def load_yaml(relative_path: str) -> dict:
    path = ROOT / relative_path
    require(path.is_file(), f"missing {relative_path}")
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


required_files = {
    "README.md",
    "LICENSE",
    "citation.bib",
    "requirements.txt",
    "assets/framework.png",
    "data/phase1_unified_clean.jsonl",
    "data/stage3_error_samples.json",
    "scripts/train_stage1_sft_v2.py",
    "scripts/train_grpo.py",
    "scripts/train_grpo_rewrite.py",
    "scripts/data_processor_phase1.py",
    "scripts/data_processor.py",
    "scripts/eval_gsm8k.py",
    "configs/stage1_sft_v2.yaml",
    "configs/stage2_grpo.yaml",
    "configs/stage3_rewrite_grpo_v2.yaml",
}
for relative_path in sorted(required_files):
    require((ROOT / relative_path).is_file(), f"missing {relative_path}")

cache_dirs = [p.relative_to(ROOT) for p in ROOT.rglob("__pycache__")]
cache_dirs += [p.relative_to(ROOT) for p in ROOT.rglob("pycache")]
require(not cache_dirs, f"cache directories must not ship: {cache_dirs}")

scannable_paths = [ROOT / "README.md"]
scannable_paths.extend((ROOT / "scripts").glob("*.py"))
scannable_paths.extend((ROOT / "configs").glob("*.yaml"))
for path in scannable_paths:
        text = path.read_text(encoding="utf-8")
        require("/home/bowyu2" not in text, f"absolute server path in {path.relative_to(ROOT)}")
        require("../.." not in text, f"parent traversal path in {path.relative_to(ROOT)}")

for path in sorted((ROOT / "scripts").glob("*.py")):
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        PROBLEMS.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")

stage1 = load_yaml("configs/stage1_sft_v2.yaml")
stage2 = load_yaml("configs/stage2_grpo.yaml")
stage3 = load_yaml("configs/stage3_rewrite_grpo_v2.yaml")

require(stage1.get("model_name") == "Qwen/Qwen2.5-3B", "Stage 1 must use Qwen2.5-3B")
require(stage1.get("train_data_path") == "data/phase1_unified_clean.jsonl", "wrong Stage-1 data path")
require(stage1.get("mask_probability") == 0.7, "wrong Stage-1 sample mask probability")
require(stage1.get("num_epochs") == 5, "wrong Stage-1 epoch count")
require(stage2.get("base_model") == f"{stage1.get('output_dir')}/final_model", "Stage-2 input does not match Stage-1 output")
require(stage2.get("train_data_path") == "data/phase1_unified_clean.jsonl", "wrong Stage-2 data path")
require(stage2.get("mask_probability") == 0.5 and stage2.get("min_mask_steps") == 3, "wrong Stage-2 masking settings")
require(stage3.get("base_model") == "models/stage2_for_superrl", "Stage 3 must default to the checkpoint used in the reported run")
require(stage3.get("train_data_path") == "data/stage3_error_samples.json", "wrong Stage-3 data path")

phase1_path = ROOT / "data/phase1_unified_clean.jsonl"
if phase1_path.is_file():
    with phase1_path.open(encoding="utf-8") as handle:
        phase1_rows = [json.loads(line) for line in handle if line.strip()]
    require(len(phase1_rows) == 6128, f"expected 6128 Stage-1/2 rows, found {len(phase1_rows)}")
    require(all({"input", "target", "metadata"} <= row.keys() for row in phase1_rows), "invalid Stage-1/2 data schema")

stage3_path = ROOT / "data/stage3_error_samples.json"
if stage3_path.is_file():
    stage3_rows = json.loads(stage3_path.read_text(encoding="utf-8"))
    require(len(stage3_rows) == 568, f"expected 568 Stage-3 rows, found {len(stage3_rows)}")
    require(all({"question", "gold_answer", "teacher_cot"} <= row.keys() for row in stage3_rows), "invalid Stage-3 data schema")

requirements = {
    re.split(r"[<>=]", line.strip(), maxsplit=1)[0].lower()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
for package in {
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "numpy",
    "pyyaml",
    "tqdm",
    "nltk",
    "sentence-transformers",
    "tensorboard",
}:
    require(package in requirements, f"missing dependency: {package}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for marker in {
    "models/stage2_for_superrl",
    "0.3 × min(step reduction, 3)",
    "original_steps × 35",
    "response_words / teacher_words",
    "RewardCalculator.calculate",
    "RewriteRewardCalculator.calculate",
}:
    require(marker in readme, f"README is missing the verified method description: {marker}")
require("format term **+0.5**" not in readme, "README contains a Stage-3 format reward not used by the code")

eval_text = (ROOT / "scripts/eval_gsm8k.py").read_text(encoding="utf-8")
for argument in ("--model_path", "--num_samples", "--output_dir", "--gpu"):
    require(argument in eval_text, f"evaluation CLI is missing {argument}")

if PROBLEMS:
    print("RELEASE CHECK FAILED")
    for problem in PROBLEMS:
        print(f"- {problem}")
    sys.exit(1)

print("RELEASE CHECK PASSED")
