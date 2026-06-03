# -*- coding: utf-8 -*-

import json
import torch
import os
import re
import random
import evaluate
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paths
from utils import *
from dataset_utils import dataset_mapping, DatasetBase
from qwen_model_wrapper import QwenModelWrapper
from llama_model_wrapper import LlamaModelWrapper
try:
    from llava_ov_model_wrapper import LlavaOVModelWrapper
except ModuleNotFoundError:
    LlavaOVModelWrapper = None
from qwen_vl_model_wrapper import QwenVLModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Idefics3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
from src.shift_encoder import AttnFFNShift, AttnApproximator, ShiftStrategy # Import necessary classes

try:
    from idefics3_model_wrapper import Idefics3ModelWrapper
except ModuleNotFoundError:
    Idefics3ModelWrapper = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
import re
import sys
from torch.utils.data import DataLoader


def _eval_internlm_strategyqa_baseline(cfg: DictConfig, dataset: DatasetBase, model):
    generation_args = {k: v for k, v in cfg.generation_args.items()}
    # InternLM2 remote_code in this environment is not compatible with the
    # current cache_position path during generation; disabling KV cache avoids
    # the one-token length mismatch while keeping this short-answer baseline usable.
    generation_args["use_cache"] = False
    batch_size = cfg.batch_size
    use_cot = bool(cfg.use_cot)
    num_shot = cfg.data.num_shot

    dataset.dataset_state = dataset._resolve_eval_state(getattr(cfg, "eval_mode", None))

    sampled_few_shot_examples = []
    if num_shot > 0 and (
        dataset.dataset_state == dataset.dataset_state.EVAL_BASELINE
        or dataset.dataset_state == dataset.dataset_state.EVAL_WITH_COT_VECTOR_ONESHOT
    ):
        num_available_examples = len(dataset._support_set)
        if num_shot > num_available_examples:
            print(
                f"Warning: Requested {num_shot} shots, but only {num_available_examples} available in training set. Using all available examples."
            )
            sampled_few_shot_examples = dataset._support_set[:]
        else:
            sampled_few_shot_examples = random.sample(list(dataset._support_set), num_shot)

    if getattr(cfg, "mode", "eval") == "generate_self_cot":
        dataset_to_load = dataset._support_set
        print(
            f"Mode is 'generate_self_cot', loading training set ({len(dataset_to_load)} samples) for Self-CoT generation."
        )
    else:
        dataset_to_load = dataset._query_set
        print(f"Mode is 'eval', loading test set ({len(dataset_to_load)} samples) for evaluation.")

    dataloader = DataLoader(
        dataset_to_load,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_to_load.collate_fn if hasattr(dataset_to_load, "collate_fn") else None,
    )

    records = []
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating StrategyQA (InternLM)"):
        questions = batch["question"]
        ground_truth_answers = batch["answer"]
        if hasattr(ground_truth_answers, "tolist"):
            ground_truth_answers = ground_truth_answers.tolist()
        ground_truth_answers = [str(item) for item in ground_truth_answers]

        predictions = []
        for question in questions:
            messages = dataset._create_qwen_chat_template(
                question=question,
                use_cot=use_cot,
                few_shot_examples=sampled_few_shot_examples,
                dataset_state=dataset.dataset_state,
            )

            system_prompt = ""
            history = []
            pending_user = None
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    pending_user = content
                elif role == "assistant" and pending_user is not None:
                    history.append((pending_user, content))
                    pending_user = None

            query = pending_user if pending_user is not None else question
            response, _ = model.model.chat(
                tokenizer=model.processor,
                query=query,
                history=history if history else None,
                streamer=None,
                meta_instruction=system_prompt,
                **generation_args,
            )
            predictions.append(response)

        for idx, pred in enumerate(predictions):
            question = questions[idx]
            ground_truth = str(ground_truth_answers[idx])
            extracted_pred = dataset.extract_answer(pred)
            is_correct = extracted_pred == ground_truth

            records.append(
                {
                    "question": question,
                    "model_output": pred,
                    "extracted_prediction": extracted_pred,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                }
            )

            if is_correct:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return records, {"accuracy": accuracy}


def _eval_internlm_commonsenseqa_baseline(cfg: DictConfig, dataset: DatasetBase, model):
    generation_args = {k: v for k, v in cfg.generation_args.items()}
    # Reuse the same workaround as StrategyQA: InternLM2 remote_code in this
    # environment can fail on generate() due to cache_position length mismatch.
    generation_args["use_cache"] = False
    batch_size = cfg.batch_size
    use_cot = bool(cfg.use_cot)
    num_shot = cfg.data.num_shot

    dataset.dataset_state = dataset._resolve_eval_state(getattr(cfg, "eval_mode", None))

    if dataset.dataset_state == dataset.dataset_state.EVAL_BASELINE and not use_cot:
        direct_answer_cap = int(getattr(cfg, "direct_answer_max_new_tokens", 8))
        generation_args["max_new_tokens"] = min(
            int(generation_args.get("max_new_tokens", direct_answer_cap)),
            direct_answer_cap,
        )

    sampled_few_shot_examples = []
    if num_shot > 0 and dataset.dataset_state == dataset.dataset_state.EVAL_BASELINE:
        num_available_examples = len(dataset._support_set)
        if num_shot > num_available_examples:
            print(
                f"Warning: Requested {num_shot} shots, but only {num_available_examples} available in training set. Using all available examples."
            )
            sampled_few_shot_examples = dataset._support_set[:]
        else:
            sampled_few_shot_examples = random.sample(list(dataset._support_set), num_shot)

    if getattr(cfg, "mode", "eval") == "generate_self_cot":
        dataset_to_load = dataset._support_set
        print(
            f"Mode is 'generate_self_cot', loading training set ({len(dataset_to_load)} samples) for Self-CoT generation."
        )
    else:
        dataset_to_load = dataset._query_set
        print(f"Mode is 'eval', loading test set ({len(dataset_to_load)} samples) for evaluation.")

    dataloader = DataLoader(
        dataset_to_load,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: dataset.collate_fn_for_eval(batch),
    )

    records = []
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating CommonsenseQA (InternLM)"):
        questions = batch["question"]
        ground_truth_answers = batch["answerKey"]
        all_options = batch["options"]

        predictions = []
        for idx, question in enumerate(questions):
            messages = dataset._create_qwen_chat_template(
                question=question,
                use_cot=use_cot,
                options=all_options[idx],
                few_shot_examples=sampled_few_shot_examples,
                dataset_state=dataset.dataset_state,
            )

            system_prompt = ""
            history = []
            pending_user = None
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    pending_user = content
                elif role == "assistant" and pending_user is not None:
                    history.append((pending_user, content))
                    pending_user = None

            query = pending_user if pending_user is not None else question
            response, _ = model.model.chat(
                tokenizer=model.processor,
                query=query,
                history=history if history else None,
                streamer=None,
                meta_instruction=system_prompt,
                **generation_args,
            )
            predictions.append(response)

        for idx, pred in enumerate(predictions):
            question = questions[idx]
            ground_truth = str(ground_truth_answers[idx])
            extracted_pred = dataset.extract_answer(pred)
            if extracted_pred is None:
                extracted_pred = random.choice(["A", "B", "C", "D", "E"])
            is_correct = extracted_pred == ground_truth

            records.append(
                {
                    "question": question,
                    "model_output": pred,
                    "extracted_prediction": extracted_pred,
                    "extracted_ground_truth": ground_truth,
                    "ground_truth": ground_truth,
                    "options": all_options[idx],
                    "is_correct": is_correct,
                }
            )

            if is_correct:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return records, {"accuracy": accuracy}


@hydra.main(config_path="config", config_name="eval_baseline.yaml", version_base=None)
def main(cfg: DictConfig):
    # initialize dataset and variables
    runname = get_expand_runname(cfg)
    is_icl = cfg.ckpt_path is None
    cfg.data.is_icl = is_icl
    
    mode = getattr(cfg, "mode", "eval")  # 获取mode参数，默认为eval

    dataset: DatasetBase = dataset_mapping[cfg.data.name](cfg.data, model_name=cfg.model_name)
    
    if mode == "generate_self_cot":
        # 对于Self-CoT生成模式，直接使用cfg中指定的output_path
        output_dir = os.path.dirname(cfg.output_path)
        os.makedirs(output_dir, exist_ok=True)
        record_dir = output_dir
        record_path = cfg.output_path
        print(f"Self-CoT generation mode. Output will be saved to {cfg.output_path}")
    else:
        # 对于常规评估模式
        if is_icl:
            if cfg.use_extracted_cot_vector:
                record_dir = os.path.join(paths.result_dir, "record")
                if cfg.use_base_vector:
                    # 如果是 base 版本，使用固定的 'base_vector' 作为文件夹名
                    folder_name = "ov_gsm8k_base_vector" if cfg.use_extracted_cot_vector_type == "licv" else "mimic_base_vector"
                elif cfg.use_extracted_cot_vector_type == "licv":
                # 新增 licv 版本处理：使用固定的 'licv_version' 作为文件夹名
                    folder_name = "ov_gsm8k_licv_vector"
                elif cfg.use_extracted_cot_vector_type == "mimic":
                    folder_name = "ov_gsm8k_mimic_vector"
                else:
                    # 如果是 mimic 或其他版本，则根据 extracted_cot_vector_path 命名
                    folder_name = os.path.basename(cfg.extracted_cot_vector_path).replace('.pt', '')
                
                # 将生成的文件夹名添加到 record_dir 中
                record_dir = os.path.join(record_dir, folder_name)
                record_root = getattr(cfg, "record_root", None)
                if record_root:
                    record_dir = record_root
                # --- 修改结束 ---

                os.makedirs(record_dir, exist_ok=True)
               
                eval_mode_suffix = ""
                if cfg.eval_mode == "EVAL_WITH_COT_VECTOR_ONESHOT":
                    eval_mode_suffix = "_oneshot"
                elif cfg.eval_mode == "EVAL_WITH_COT_VECTOR_DIRECT_Q":
                    eval_mode_suffix = "_direct_q"

                if cfg.use_base_vector:
                    # Format layer string for filename based on its content, assuming it's a string
                    layers_for_filename = str(cfg.only_shift_at_layer)
                    if '[' in layers_for_filename and ']' in layers_for_filename: # Heuristic for list string
                        layers_for_filename = layers_for_filename.replace('[', '').replace(']', '').replace(', ', '_')

                    record_path = os.path.join(record_dir,
                                               f"use_base_vector_layers_{layers_for_filename}{eval_mode_suffix}_{cfg.static_mu_value}.json")
                else:
                    # Format layer string for filename based on its content, assuming it's a string
                    layers_for_filename = str(cfg.only_shift_at_layer)
                    if '[' in layers_for_filename and ']' in layers_for_filename: # Heuristic for list string
                        layers_for_filename = layers_for_filename.replace('[', '').replace(']', '').replace(', ', '_')

                    record_path = os.path.join(record_dir, f"{cfg.use_extracted_cot_vector_type}_layers_{layers_for_filename}{eval_mode_suffix}_{cfg.static_mu_value}.json")
            else:
                record_dir = os.path.join(paths.result_dir, "record", runname)
                # Construct base filename parts
                file_name_parts = [
                    f"{cfg.data.num_shot}shot",
                ]
                record_filename = "_".join(file_name_parts) + ".json"
                record_path = os.path.join(record_dir, record_filename)
        else:
            record_dir = os.path.join(paths.result_dir, "record", runname)
            if not os.path.exists(cfg.ckpt_path) and not cfg.use_extracted_cot_vector: # Add check for extracted vector
                raise FileNotFoundError(f"Checkpoint path {cfg.ckpt_path} not found.")

            epoch = re.findall(r"\d+", os.path.basename(cfg.ckpt_path))
            if len(epoch) != 1:
                raise ValueError(
                    f"Invalid checkpoint path {cfg.ckpt_path}. It should contain a single number in basename for epoch."
                )
            epoch = int(epoch[0])
            record_path = os.path.join(record_dir, f"epoch-{epoch}.json")

    custom_record_root = getattr(cfg, "record_root", None)
    if custom_record_root and mode != "generate_self_cot":
        record_dir = custom_record_root
        os.makedirs(record_dir, exist_ok=True)
        record_path = os.path.join(record_dir, os.path.basename(record_path))
    else:
        os.makedirs(record_dir, exist_ok=True)


    print(f"DEBUG: The value of cfg.only_shift_at_layer is: {cfg.only_shift_at_layer} (type: {type(cfg.only_shift_at_layer)})")

    # 构建 record_path 的代码...

    print(f"DEBUG: The constructed record_path is: {record_path}")
    
    if cfg.resume:
        if os.path.exists(record_path):
            print(f"Found exist record {record_path}, skip...")
            return

    device = get_runtime_device()

    model_name_lower = cfg.model_name.lower()

    if "llama" in model_name_lower:
        model_id, model_path = build_model(cfg)
        model = LlamaModelWrapper(
            model_root=model_path,  # Pass the local path as model_root
            processor_class=AutoTokenizer,  # Pass the class itself
            model_class=AutoModelForCausalLM,  # Pass the class itself
            support_models=[model_id],  # Pass original HF ID for tracking/validation
            local_files_only=True,  # Ensure local files are used
            torch_dtype=eval(cfg.dtype),  # Pass torch_dtype to ModelBase
            processor_args={
                "padding_side": "left",
                "use_fast":False
                },  # Add this line to set padding_side
            model_name=cfg.model_name  # Pass model_name to LlamaModelWrapper
        ).to(device)
        if model.processor.pad_token is None:
            model.processor.pad_token = model.processor.eos_token

    elif "qwen2.5-vl" in model_name_lower:
        # build_model 对于新模型通常返回 (model_id, model_path)
        model_id, model_path = build_model(cfg) 
        model = QwenVLModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=Qwen2_5_VLForConditionalGeneration,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name
        ).to(device)
        dataset.model_processor = model.processor
    elif "qwen2.5" in model_name_lower or "internlm" in model_name_lower:
        model_id, model_path = build_model(cfg)
        model = QwenModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModelForCausalLM,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={
                "padding_side": "left",
                "use_fast": False,
            },
            model_name=cfg.model_name,
        ).to(device)
        if model.processor.pad_token is None:
            model.processor.pad_token = model.processor.eos_token

    elif "Llava-OneVision-1.5-8B" in cfg.model_name.lower():
        model_id, model_path = build_llava_model_info(cfg)
        model = LlavaOVModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=LlavaOnevisionForConditionalGeneration,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True, "trust_remote_code": True},
            model_name=cfg.model_name,
        ).to(device)

        # 多模态数据集通常需要 processor
        dataset.model_processor = model.processor
    elif "idefics3" in cfg.model_name.lower():
        if Idefics3ModelWrapper is None:
            raise ModuleNotFoundError(
                "idefics3_model_wrapper is required only for Idefics3 runs. "
                "This baseline does not need it, but an Idefics3 experiment would."
            )
        model_id, model_path = build_idefics3_model_info(cfg)
        model_args = {}
        if torch.cuda.is_available():
            # Idefics3 is more reliable when loaded directly onto the visible GPU
            # via device_map than when materialized on CPU and moved afterward.
            model_args["device_map"] = "cuda:0"
            model_args["low_cpu_mem_usage"] = True
        model = Idefics3ModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=Idefics3ForConditionalGeneration,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args=model_args,
            model_name=cfg.model_name,
        )
        dataset.model_processor = model.processor

    else:
        # 对于其他模型，如果是元组则取第一个元素
        result = build_model(cfg)
        if isinstance(result, tuple):
            model = result[0].to(device, eval(cfg.dtype))
        else:
            model = result.to(device, eval(cfg.dtype))

    # 如果 cfg.ckpt_path 存在，那么它就不是 ICL baseline
    is_icl = cfg.ckpt_path is None # 恢复原始逻辑，根据 ckpt_path 判断是否为 ICL

    encoder = None # Initialize encoder to None
    hooks = {} # Initialize hooks dictionary

    if cfg.use_extracted_cot_vector:
        if not cfg.extracted_cot_vector_path or not os.path.exists(cfg.extracted_cot_vector_path):
            raise FileNotFoundError(f"Extracted CoT vector file not found: {cfg.extracted_cot_vector_path}")
        
        print(f"Loading extracted CoT vectors from: {cfg.extracted_cot_vector_path}")
        loaded_vectors = torch.load(cfg.extracted_cot_vector_path, map_location=device, weights_only=False)

        # Determine strategies based on presence of vectors and multi_head_attn_strategy
        # Build strategy strings for BaseHookEncoder's eval()
        attn_strat_str = "ShiftStrategy(0)"
        if cfg.peft.name == "licv" and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
            print("The shift encoder version conflict! Now we change it to LICV version.")
            loaded_vectors["encoder_type"]._target_ = "src.shift_encoder.AttnFFNShift"

        if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
            if loaded_vectors["attn_cot_vector"] is not None:
                attn_strat_str = "ShiftStrategy.VECTOR_SHIFT"
                # Now loaded_vectors["encoder_type"] is a DictConfig, so compare its _target_ attribute
                if loaded_vectors["multi_head_attn_strategy"] and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                    attn_strat_str += " | ShiftStrategy.MULTI_HEAD"

                attn_strat_str += " | ShiftStrategy.LEARNABLE_SHIFT_SCALE"
                
            # NEW: Add STATIC_MU_FROM_CONFIG to strategy if use_static_mu is enabled
            if getattr(cfg, "use_static_mu", False) and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                attn_strat_str += " | ShiftStrategy.STATIC_MU_FROM_CONFIG"

            print(f"Attn_strat_str: {attn_strat_str}")
            print(f"multi_head_attn_strategy: {loaded_vectors['multi_head_attn_strategy']}")
            print("loaded_vectors 的类型:", type(loaded_vectors))
            print("loaded_vectors 的键:", list(loaded_vectors.keys()))
            print("attn_cot_vector 的 shape:", loaded_vectors["attn_cot_vector"].shape)
            ########################################

            # print("##########")

        ffn_strat_str = "ShiftStrategy.RECORD_HIDDEN_STATES"
        if loaded_vectors["ffn_cot_vector"] is not None:
            # Check the encoder type to determine if FFN shift should be activated
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnFFNShift":
                ffn_strat_str = "ShiftStrategy.VECTOR_SHIFT"
            # For AttnApproximator, FFN shift remains off as per previous user request.

        # This handles the _target_ and other default parameters correctly.
        only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
        if isinstance(only_shift_at_layer, str): # If it's a string, try to parse it as a list
            try:
                # Attempt to parse as JSON list or other JSON type
                parsed_value = json.loads(only_shift_at_layer)
                # Only update if parsing was successful and it's a list or a simple value (like int)
                if isinstance(parsed_value, (list, int, type(None))):
                    only_shift_at_layer = parsed_value
            except json.JSONDecodeError:
                # If it's not a valid JSON string, keep it as is (e.g., 'null' or a single integer string)
                pass
        
        encoder = hydra.utils.instantiate(
            loaded_vectors["encoder_type"],  # Pass the full DictConfig here
            lmm=model,
            attn_strategy=attn_strat_str, # Pass the string representation
            ffn_strategy=ffn_strat_str,   # Pass the string representation
            shift_scale_init_value=cfg.static_shift_scale, # Ensure scale is 1.0 for static vectors
            # NEW: Pass static_mu_value to encoder if enabled
            static_mu_value=getattr(cfg, "static_mu_value", None) if getattr(cfg, "use_static_mu", False) else None,
            # NEW: Pass only_shift_at_layer to encoder if enabled
            only_shift_at_layer=only_shift_at_layer,
            _recursive_=False  # Prevent deep instantiation if not needed, for clarity
        ).to(device, eval(cfg.dtype))

        # Assign the loaded vectors to the encoder's parameters
        with torch.no_grad():  # Ensure these assignments don't get recorded for gradients
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnFFNShift":
                if cfg.use_base_vector:
                    encoder.ffn_shift.data = loaded_vectors["ffn_hs_vector"].to(device).to(model.model.dtype)
                else:
                    encoder.ffn_shift.data = loaded_vectors["ffn_cot_vector"].to(device).to(model.model.dtype)
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                encoder.attn_shift.data = loaded_vectors["attn_cot_vector"].to(device).to(model.model.dtype)

        encoder.eval()
        hooks = encoder.register_shift_hooks()

    elif not is_icl: # This is the original logic for loading trained checkpoints
        encoder = hydra.utils.instantiate(cfg.encoder.cls, lmm=model).to(
            device, eval(cfg.dtype)
        )

        print(f"Loading from pretrained: {cfg.ckpt_path}")
        load_from_pretrained(cfg.ckpt_path, model, encoder)
        encoder.eval()
        hooks = encoder.register_shift_hooks()

    model.eval()

    try:
        if "internlm" in model_name_lower and not cfg.use_extracted_cot_vector and is_icl:
            if cfg.data.name == "strategyqa":
                result, eval_result = _eval_internlm_strategyqa_baseline(cfg, dataset, model)
            elif cfg.data.name == "commonsenseqa":
                result, eval_result = _eval_internlm_commonsenseqa_baseline(cfg, dataset, model)
            else:
                result, eval_result = dataset.eval(cfg, model)
        else:
            # Pass mode to dataset.eval
            result, eval_result = dataset.eval(cfg, model)
        print(f"Evaluation result for {runname}: {eval_result}")

        if mode == "generate_self_cot":
            # Save all generated results to .jsonl
            full_output_path = cfg.output_path
            with open(full_output_path, 'w', encoding='utf-8') as f:
                for record in result: # 'result' here is the list of records from dataset.eval
                    # Ensure the 'model_output' (self_cot) and 'is_correct' fields are present
                    # And 'question', 'extracted_ground_truth' (gt_answer)
                    # Map to the desired self_cot data format: question, self_cot, is_correct, gt_answer
                    self_cot_record = {
                        "question": record["question"],
                        "self_cot": record["model_output"], # Model output is the generated CoT
                        "is_correct": record["is_correct"],
                        "gt_answer": record["extracted_ground_truth"] # Ground truth answer
                    }
                    f.write(json.dumps(self_cot_record, ensure_ascii=False) + '\n')
            print(f"All generated Self-CoT saved to {full_output_path}")

            # Save only correct samples to _correct_only.jsonl
            correct_output_path = cfg.output_path.replace('.json', '_correct_only.json')
            with open(correct_output_path, 'w', encoding='utf-8') as f:
                for record in result:
                    if record["is_correct"]:
                        self_cot_record = {
                            "question": record["question"],
                            "self_cot": record["model_output"],
                            "is_correct": record["is_correct"],
                            "gt_answer": record["extracted_ground_truth"]
                        }
                        f.write(json.dumps(self_cot_record, ensure_ascii=False) + '\n')
            print(f"Correct Self-CoT samples saved to {correct_output_path}")

        else:
            # Regular evaluation saving logic
            Path(record_path).touch() # Ensure file exists for evaluate.save if it's new
        config = {"eval_args": OmegaConf.to_container(cfg, resolve=True)}
        if os.path.exists(os.path.join(record_dir, "config.json")):
            with open(os.path.join(record_dir, "config.json")) as f:
                config["train_args"] = json.load(f)

        evaluate.save(
            record_path,
            **config,
            eval_result=eval_result,
            records=result,
        )

    finally:
        if encoder: # Only remove hooks if encoder was instantiated
            encoder.remove_hooks(hooks)
        if os.path.exists(record_path) and os.path.getsize(record_path) == 0:
            os.remove(record_path)


if __name__ == "__main__":
    main()
