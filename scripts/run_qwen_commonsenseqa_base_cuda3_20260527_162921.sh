#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data1/wzy/cot-mimic:/data1/wzy/cot-mimic/src
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp

ROOT=/data1/wzy/cot-mimic
VECTOR=$ROOT/results/static_pipeline/qwen2.5-7b-instruct_commonsenseqa_ffn_20260527_141517/qwen2.5-7b-instruct_commonsenseqa_ffn_20260527_141517.pt
RUN_TAG=qwen2.5-7b-instruct_commonsenseqa_base_20260527_162921
RUN_DIR=$ROOT/results/static_pipeline/$RUN_TAG
LOG_DIR=$RUN_DIR/logs
RECORD_DIR=$RUN_DIR/records

mkdir -p "$HF_HOME/datasets" "$HF_HOME/transformers" "$RUN_DIR" "$LOG_DIR" "$RECORD_DIR"

: > $LOG_DIR/03_eval_layerwise.log
for layer in $(seq 0 27); do
  echo "===== layer_${layer} =====" >> $LOG_DIR/03_eval_layerwise.log
  stdbuf -oL -eL /home/wzy/anaconda3/envs/licv/bin/python \
    $ROOT/src/eval_base.py \
    --config-name eval_base \
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
    use_extracted_cot_vector_type=licv \
    extracted_cot_vector_path=$VECTOR \
    use_base_vector=True \
    generation_args.max_new_tokens=8 \
    +direct_answer_max_new_tokens=8 \
    +peft=licv \
    +encoder=licv \
    > >(tee -a $LOG_DIR/03_eval_layerwise.log) 2> >(tee -a $LOG_DIR/03_eval_layerwise.log >&2)
done
