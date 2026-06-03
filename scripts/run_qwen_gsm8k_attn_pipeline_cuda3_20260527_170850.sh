#!/usr/bin/env bash
set -euo pipefail

cd /data1/wzy/cot-mimic

export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/data1/wzy/cot-mimic:/data1/wzy/cot-mimic/src

/home/wzy/anaconda3/envs/licv/bin/python /data1/wzy/cot-mimic/scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset gsm8k \
  --method attn \
  --devices 3 \
  --run-tag qwen2.5-7b-instruct_gsm8k_attn_20260527_170850
