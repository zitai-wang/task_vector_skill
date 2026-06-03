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

/home/wzy/anaconda3/envs/licv/bin/python /data1/wzy/cot-mimic/scripts/run_eval_layers_mimic.py \
  model_name=qwen2.5-7b-instruct \
  data.name=gsm8k \
  data.num_shot=0 \
  data.num_query_samples=1319 \
  ++data.source=/data/share/datasets/gsm8k \
  devices=0 \
  batch_size=8 \
  resume=False \
  extracted_cot_vector_path=/data1/wzy/cot-mimic/results/static_pipeline/qwen2.5-7b-instruct_gsm8k_attn_20260527_170850_20260527_183104/qwen2.5-7b-instruct_gsm8k_attn_20260527_170850_20260527_183104.pt \
  +record_dir_tag=qwen2.5-7b-instruct_gsm8k_attn_20260527_170850_20260527_183104 \
  ++record_root=/data1/wzy/cot-mimic/results/static_pipeline/qwen2.5-7b-instruct_gsm8k_attn_20260527_170850_20260527_183104/records
