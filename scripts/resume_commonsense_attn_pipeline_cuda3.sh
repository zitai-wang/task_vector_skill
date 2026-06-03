#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data1/wzy/cot-mimic:/data1/wzy/cot-mimic/src
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp

ROOT=/data1/wzy/cot-mimic
SRC_RUN_DIR=$ROOT/results/static_pipeline/qwen2.5-7b-instruct_commonsenseqa_20260526_201516
RUN_TAG=qwen2.5-7b-instruct_commonsenseqa_20260527_112252
RUN_DIR=$ROOT/results/static_pipeline/$RUN_TAG
VECTOR=$RUN_DIR/$RUN_TAG.pt
LOG_DIR=$RUN_DIR/logs
RECORD_DIR=$RUN_DIR/records

mkdir -p "$HF_HOME/datasets" "$HF_HOME/transformers" "$LOG_DIR" "$RECORD_DIR"

cp -f "$SRC_RUN_DIR/self_cot_data.json" "$RUN_DIR/self_cot_data.json"
cp -f "$SRC_RUN_DIR/self_cot_data_correct_only.json" "$RUN_DIR/self_cot_data_correct_only.json"

if [[ ! -f "$VECTOR" ]]; then
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
    > $LOG_DIR/02_extract_cuda3.log 2>&1
fi

: > $LOG_DIR/03_eval_layerwise_cuda3.log
for layer in $(seq 0 27); do
  echo "===== layer_${layer} =====" >> $LOG_DIR/03_eval_layerwise_cuda3.log
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
    >> $LOG_DIR/03_eval_layerwise_cuda3.log 2>&1
done
