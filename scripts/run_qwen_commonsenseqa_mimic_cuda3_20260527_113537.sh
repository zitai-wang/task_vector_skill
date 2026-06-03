#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data1/wzy/cot-mimic:/data1/wzy/cot-mimic/src
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp

ROOT=/data1/wzy/cot-mimic
RUN_TAG=qwen2.5-7b-instruct_commonsenseqa_20260527_113537
RUN_DIR=$ROOT/results/static_pipeline/$RUN_TAG
LOG_DIR=$RUN_DIR/logs
RECORD_DIR=$RUN_DIR/records
VECTOR=$RUN_DIR/$RUN_TAG.pt

mkdir -p "$HF_HOME/datasets" "$HF_HOME/transformers" "$RUN_DIR" "$LOG_DIR" "$RECORD_DIR"

stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
  $ROOT/src/generate_self_cot.py \
  --config-name generate_self_cot \
  model_name=qwen2.5-7b-instruct \
  data.name=commonsenseqa \
  data.num_query_samples=500 \
  ++data.source=/data/share/commonsenceqa/ \
  devices=0 \
  output_path=$RUN_DIR/self_cot_data.json \
  max_samples=500 \
  batch_size=8 \
  > $LOG_DIR/01_generate.log 2>&1

stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
  $ROOT/src/extract_cot_vector.py \
  --config-name extract_cot_vector \
  model_name=qwen2.5-7b-instruct \
  data.name=commonsenseqa \
  data.num_query_samples=500 \
  ++data.source=/data/share/commonsenceqa/ \
  +devices=0 \
  data.use_self_cot=True \
  data.self_cot_path=$RUN_DIR/self_cot_data_correct_only.json \
  output_path=$VECTOR \
  batch_size=1 \
  > $LOG_DIR/02_extract.log 2>&1

: > $LOG_DIR/03_eval_layerwise.log
for layer in $(seq 0 27); do
  echo "===== layer_${layer} =====" >> $LOG_DIR/03_eval_layerwise.log
  stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
    $ROOT/src/eval.py \
    --config-name eval_attn \
    model_name=qwen2.5-7b-instruct \
    data.name=commonsenseqa \
    data.num_query_samples=1221 \
    ++data.source=/data/share/commonsenceqa/ \
    devices=0 \
    batch_size=8 \
    only_shift_at_layer=${layer} \
    resume=False \
    +record_root=$RECORD_DIR \
    use_extracted_cot_vector=True \
    use_extracted_cot_vector_type=mimic \
    extracted_cot_vector_path=$VECTOR \
    generation_args.max_new_tokens=8 \
    direct_answer_max_new_tokens=8 \
    >> $LOG_DIR/03_eval_layerwise.log 2>&1
done
