#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data1/wzy/cot-mimic:/data1/wzy/cot-mimic/src
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp

ROOT=/data1/wzy/cot-mimic
RUN_DIR=$ROOT/results/static_pipeline/qwen2.5-7b-instruct_gsm8k_20260526_193601
VECTOR=$ROOT/results/average_cot_vectors/qwen2.5-7b-instruct_gsm8k_20260526_193601.pt

mkdir -p "$HF_HOME/datasets" "$HF_HOME/transformers"

stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
  $ROOT/src/extract_cot_vector.py \
  --config-name extract_cot_vector \
  model_name=qwen2.5-7b-instruct \
  data.name=gsm8k \
  data.num_query_samples=500 \
  ++data.source=/data/share/datasets/gsm8k \
  +devices=0 \
  data.use_self_cot=True \
  data.self_cot_path=$RUN_DIR/self_cot_data_correct_only.json \
  output_path=$VECTOR \
  batch_size=1 \
  > $RUN_DIR/logs/02_extract.log 2>&1

: > $RUN_DIR/logs/03_eval_layerwise.log
for layer in $(seq 0 27); do
  echo "===== layer_${layer} =====" >> $RUN_DIR/logs/03_eval_layerwise.log
  stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
    $ROOT/src/eval.py \
    --config-name eval_ffn \
    model_name=qwen2.5-7b-instruct \
    data.name=gsm8k \
    data.num_query_samples=1319 \
    ++data.source=/data/share/datasets/gsm8k \
    devices=0 \
    batch_size=8 \
    only_shift_at_layer=${layer} \
    resume=False \
    use_extracted_cot_vector=True \
    extracted_cot_vector_path=$VECTOR \
    >> $RUN_DIR/logs/03_eval_layerwise.log 2>&1
done
