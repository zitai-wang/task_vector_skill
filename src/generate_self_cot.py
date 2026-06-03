#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image
import torch
import re
from omegaconf import DictConfig, OmegaConf
import hydra

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import build_model
from internvl_model_wrapper import InternVLModelWrapper
from qwen_model_wrapper import QwenModelWrapper
from llama_model_wrapper import LlamaModelWrapper
from qwen_vl_model_wrapper import QwenVLModelWrapper
try:
    from llava_ov_model_wrapper import LlavaOVModelWrapper
except ModuleNotFoundError:
    LlavaOVModelWrapper = None
try:
    from eval_scienceqa_internvl_baseline import (
        ensure_generation_support as ensure_internvl_generation_support,
        patch_prepare_inputs_for_generation as patch_internvl_prepare_inputs_for_generation,
        generate_image_batch as generate_internvl_image_batch,
        generate_text_batch as generate_internvl_text_batch,
    )
except ModuleNotFoundError:
    ensure_internvl_generation_support = None
    patch_internvl_prepare_inputs_for_generation = None
    generate_internvl_image_batch = None
    generate_internvl_text_batch = None

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from dataset_utils.mmlu import MMLUProDataset
from dataset_utils.math_dataset import MATHDataset
from src.dataset_utils.mmmu import MMMUDataset
from src.dataset_utils.mathvista import MathVistaDataset
from src.dataset_utils.mathvision import MathVisionDataset
from src.dataset_utils.scienceqa import ScienceQADataset
from dataset_utils.gsm8k import GSM8KDataset
from src.dataset_utils.commonsenceqa import CommonsenseQADataset
from src.dataset_utils.strategyqa import StrategyQADataset


def build_self_cot_instruction(dataset_name: str) -> str:
    if dataset_name == "strategyqa":
        return (
            "You are a helpful and precise assistant for answering yes/no questions. "
            "Please reason step by step, and end with a final answer of either True or False. "
            "If you use \\boxed{}, put only True or False inside."
        )
    return "Please reason step by step, and put your final answer within \\boxed{}."

def generate_self_cot_for_sample(model, question: str, generation_args: Dict, image=None, dataset_name: str = "") -> str:
    """
    完整的多模态 Self-CoT 生成函数。
    适配 Qwen2.5-VL 和 LLaVA-OneVision。
    """
    # 1. 构造统一的推理指令
    instruction = build_self_cot_instruction(dataset_name)
    normalized_question = str(question).strip()
    if normalized_question.startswith("Question:") or normalized_question.startswith("Context:"):
        full_prompt = f"{instruction}\n{normalized_question}"
    else:
        full_prompt = f"{instruction}\nQuestion: {normalized_question}"

    try:
        valid_image = image if (image is not None and hasattr(image, 'size')) else None

        if isinstance(image, Image.Image):
            valid_image = image
        elif isinstance(image, (list, tuple)) and image and all(isinstance(img, Image.Image) for img in image):
            valid_image = list(image)
        else:
            valid_image = None

        llava_types = (LlavaOVModelWrapper,) if LlavaOVModelWrapper is not None else ()
        if isinstance(model, (QwenVLModelWrapper, *llava_types)):
            response = model.generate(
                [full_prompt], 
                images=[valid_image] if valid_image is not None else None,
                **generation_args
            )
        elif isinstance(model, InternVLModelWrapper):
            if generate_internvl_image_batch is None or generate_internvl_text_batch is None:
                raise ModuleNotFoundError(
                    "InternVL helper functions are unavailable because "
                    "`eval_scienceqa_internvl_baseline` is missing."
                )
            tokenizer = model.processor
            internvl_model = model.model
            device = next(internvl_model.language_model.parameters()).device
            dtype = next(internvl_model.parameters()).dtype

            if valid_image is not None:
                response = generate_internvl_image_batch(
                    model=internvl_model,
                    tokenizer=tokenizer,
                    images=[valid_image],
                    questions=[full_prompt],
                    generation_args=generation_args,
                    input_size=448,
                    max_num_tiles=12,
                    device=device,
                    dtype=dtype,
                )
            else:
                response = generate_internvl_text_batch(
                    model=internvl_model,
                    tokenizer=tokenizer,
                    questions=[full_prompt],
                    generation_args=generation_args,
                )
             
        else:
            # --- 针对单模态模型（Qwen2.5-Math / Llama3）的方案 ---
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
            
            # 使用模型自带的 processor (Tokenizer) 套用模板
            prompt = model.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            response = model.generate([prompt], **generation_args)

        # 2. 统一结果提取
        # --- 替换开始 ---
        if response is None:
            return ""
        if isinstance(response, list):
            # 确保列表不为空且第一个元素不是 None
            if len(response) > 0 and response[0] is not None:
                return str(response[0]).strip()
            return ""
        return str(response).strip()
    
    
            
    except Exception as e:
        # 这里会捕获并打印具体的报错，方便我们进一步排查 Wrapper 内部逻辑
        print(f"Inner generation error during inference: {e}")
        return ""

def generate_self_cot_for_batch(
    model,
    questions: List[str],
    generation_args: Dict,
    images: List[Any] = None,
    dataset_name: str = "",
) -> List[str]:
    """
    批量生成 Self-CoT。
    当前优先适配：
    - 纯文本任务（gsm8k / math / mmlu）
    - 也兼容 VL wrapper 的 batch 接口
    """
    try:
        if not questions:
            return []

        instruction = build_self_cot_instruction(dataset_name)

        # 多模态模型分支
        llava_types = (LlavaOVModelWrapper,) if LlavaOVModelWrapper is not None else ()
        if isinstance(model, (QwenVLModelWrapper, *llava_types)):
            prompts = []
            for q in questions:
                normalized_question = str(q).strip()
                if normalized_question.startswith("Question:") or normalized_question.startswith("Context:"):
                    prompts.append(f"{instruction}\n{normalized_question}")
                else:
                    prompts.append(f"{instruction}\nQuestion: {normalized_question}")

            valid_images = None
            if images is not None:
                valid_images = []
                for img in images:
                    if isinstance(img, Image.Image):
                        valid_images.append(img)
                    elif isinstance(img, (list, tuple)) and img and all(isinstance(x, Image.Image) for x in img):
                        valid_images.append(list(img))
                    else:
                        valid_images.append(None)

            responses = model.generate(
                prompts,
                images=valid_images,
                **generation_args
            )
        elif isinstance(model, InternVLModelWrapper):
            if generate_internvl_image_batch is None or generate_internvl_text_batch is None:
                raise ModuleNotFoundError(
                    "InternVL helper functions are unavailable because "
                    "`eval_scienceqa_internvl_baseline` is missing."
                )
            prompts = []
            for q in questions:
                normalized_question = str(q).strip()
                if normalized_question.startswith("Question:") or normalized_question.startswith("Context:"):
                    prompts.append(f"{instruction}\n{normalized_question}")
                else:
                    prompts.append(f"{instruction}\nQuestion: {normalized_question}")

            tokenizer = model.processor
            internvl_model = model.model
            device = next(internvl_model.language_model.parameters()).device
            dtype = next(internvl_model.parameters()).dtype
            input_size = 448
            max_num_tiles = 12

            valid_images = []
            if images is None:
                valid_images = [None] * len(prompts)
            else:
                for img in images:
                    if isinstance(img, Image.Image):
                        valid_images.append(img)
                    else:
                        valid_images.append(None)

            responses = [None] * len(prompts)
            image_indices = [i for i, img in enumerate(valid_images) if img is not None]
            text_indices = [i for i, img in enumerate(valid_images) if img is None]

            if image_indices:
                image_responses = generate_internvl_image_batch(
                    model=internvl_model,
                    tokenizer=tokenizer,
                    images=[valid_images[i] for i in image_indices],
                    questions=[prompts[i] for i in image_indices],
                    generation_args=generation_args,
                    input_size=input_size,
                    max_num_tiles=max_num_tiles,
                    device=device,
                    dtype=dtype,
                )
                for idx, resp in zip(image_indices, image_responses):
                    responses[idx] = resp

            if text_indices:
                text_responses = generate_internvl_text_batch(
                    model=internvl_model,
                    tokenizer=tokenizer,
                    questions=[prompts[i] for i in text_indices],
                    generation_args=generation_args,
                )
                for idx, resp in zip(text_indices, text_responses):
                    responses[idx] = resp

        # 纯文本模型分支
        else:
            prompts = []
            for q in questions:
                messages = [
                    {
                        "role": "system",
                        "content": instruction
                    },
                    {
                        "role": "user",
                        "content": q
                    }
                ]

                prompt = model.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)

            responses = model.generate(prompts, **generation_args)

        if responses is None:
            return [""] * len(questions)

        if not isinstance(responses, list):
            responses = [responses]

        cleaned = []
        for r in responses:
            if r is None:
                cleaned.append("")
            else:
                s = str(r).strip()
                if s == "None":
                    s = ""
                cleaned.append(s)

        # 防止 wrapper 返回数量不一致
        if len(cleaned) < len(questions):
            cleaned.extend([""] * (len(questions) - len(cleaned)))
        elif len(cleaned) > len(questions):
            cleaned = cleaned[:len(questions)]

        return cleaned

    except Exception as e:
        print(f"Inner batch generation error during inference: {e}")
        return [""] * len(questions)

def extract_answer_from_response(response: str) -> str:
    """Extract numerical answer from model response."""
    # Look for \boxed{} pattern
    import re
    box_pattern = r"\\boxed{(.+)}"
    match = re.search(box_pattern, response)
    if match:
        return match.group(1).strip()
    
    # Look for \boxed{} pattern (without escape)
    box_pattern2 = r"\boxed{(.+)}"
    match = re.search(box_pattern2, response)
    if match:
        return match.group(1).strip()
    
    # Look for #### pattern
    hash_pattern = r"#### (.*)"
    match = re.search(hash_pattern, response)
    if match:
        return match.group(1).strip()
    
    # Try to extract any number at the end
    number_pattern = r"(\d+(?:\.\d+)?)\s*$"
    match = re.search(number_pattern, response.strip())
    if match:
        return match.group(1).strip()
    
    return response.strip()

def is_correct_answer(predicted_raw, ground_truth, sample=None):
    if not predicted_raw:
        return False

    # 1. 提取 \boxed{} 内的内容
    box_pattern = r"\\boxed\{([^\}]+)\}"
    match = re.search(box_pattern, str(predicted_raw))
    if match:
        extracted = match.group(1).strip()
    else:
        # 兜底逻辑：取最后一行
        lines = [l.strip() for l in str(predicted_raw).split('\n') if l.strip()]
        extracted = lines[-1] if lines else str(predicted_raw).strip()
    
    # 2. 基础清理
    extracted = re.sub(r"\\text\{([^\}]+)\}", r"\1", extracted) # 清理 \text{...}
    extracted = extracted.replace("\\", "").replace("(", "").replace(")", "").strip()
    
    gt = str(ground_truth).strip().lower()
    pred = extracted.lower()

    # 3. 核心：处理选项映射 (解决 A -> Yes)
    if sample and "choices" in sample:
        choices = [str(c).strip().lower() for c in sample["choices"]]
        
        # 情况 A: 模型输出了选项字母 (如 "a")
        if len(pred) == 1 and pred.isalpha():
            idx = ord(pred.upper()) - ord('A')
            if 0 <= idx < len(choices):
                # 检查该字母对应的选项内容是否等于 GT，或者该字母就是 GT
                if choices[idx] == gt or pred == gt:
                    return True
        
        # 情况 B: 模型输出了选项内容 (如 "yes")，而 GT 是字母 (如 "A")
        if gt in "abcd" and len(gt) == 1:
            gt_idx = ord(gt.upper()) - ord('A')
            if gt_idx < len(choices) and pred == choices[gt_idx]:
                return True

    # 4. 数值/字面直接匹配
    # 移除末尾句号或单位干扰
    pred_clean = re.sub(r"[^a-zA-Z0-9.\u4e00-\u9fa5]", "", pred)
    gt_clean = re.sub(r"[^a-zA-Z0-9.\u4e00-\u9fa5]", "", gt)
    
    if pred_clean == gt_clean or gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    # 5. 数值模糊匹配 (4.0 == 4)
    try:
        def extract_num(s):
            res = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            return float(res[0]) if res else None
        p_val, g_val = extract_num(pred), extract_num(gt)
        if p_val is not None and g_val is not None and abs(p_val - g_val) < 1e-5:
            return True
    except:
        pass

    return False

@hydra.main(config_path="config", config_name="generate_self_cot_gsm8k.yaml", version_base=None)
def main(cfg: DictConfig):
    print(f"Starting Self-CoT generation for {cfg.data.name} dataset")
    print(f"Model: {cfg.model_name}")
    print(f"Output path: {cfg.output_path}")
    print(f"Max samples: {cfg.max_samples}")
    use_images = bool(getattr(cfg.data, "use_images", True))
    print(f"Use images: {use_images}")
    
    # Build model
    if "internvl" in cfg.model_name.lower():
        if (
            ensure_internvl_generation_support is None
            or patch_internvl_prepare_inputs_for_generation is None
        ):
            raise ModuleNotFoundError(
                "InternVL generation helpers are unavailable because "
                "`eval_scienceqa_internvl_baseline` is missing."
            )
        internvl_hf_id, model_path = build_model(cfg)
        model = InternVLModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModel,
            support_models=[internvl_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left", "use_fast": False},
            model_args={},
            model_name=cfg.model_name,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        if model.processor.pad_token is None:
            model.processor.pad_token = model.processor.eos_token
        model.processor.padding_side = "left"
        ensure_internvl_generation_support(model.model.language_model)
        patch_internvl_prepare_inputs_for_generation(model.model.language_model)
    elif "vl" in cfg.model_name.lower():
        try:
            qwen_hf_id, model_path = build_model(cfg)
        except:
            # Fallback if build_model fails for VL
            qwen_hf_id = "Qwen/Qwen2.5-VL-7B-Instruct"
            model_path = "/data/share/Qwen2.5-VL-7B-Instruct/" # Default or from config

        model = QwenVLModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=Qwen2_5_VLForConditionalGeneration,
            # model_class=AutoModelForCausalLM,
            # model_class=AutoModel,
            support_models=[qwen_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"min_pixels": 256*28*28, "max_pixels": 1280*28*28}, # Example args
            # model_args={"attn_implementation": "flash_attention_2", "trust_remote_code": True},
            model_args={"trust_remote_code": True},
            model_name=cfg.model_name
        )
    elif "llava" in cfg.model_name.lower():
        if LlavaOVModelWrapper is None:
            raise ModuleNotFoundError(
                "llava_ov_model_wrapper is required only for LLaVA runs. "
                "Your current Qwen pipeline can run without it."
            )
        from utils import build_llava_model_info
        llava_hf_id, model_path = build_llava_model_info(cfg)
        
        from transformers import LlavaOnevisionForConditionalGeneration, SiglipVisionConfig, Qwen2Config
        import transformers.models.llava_onevision.configuration_llava_onevision as lo_config
        
        # Proxy class to handle parameter renaming for SigLIP (rice_vit)
        class RiceVitConfigProxy:
            def __new__(cls, **kwargs):
                # Remap keys from rice_vit format to SiglipVisionConfig format
                if "num_heads" in kwargs:
                    kwargs["num_attention_heads"] = kwargs.pop("num_heads")
                if "embed_dim" in kwargs:
                    kwargs["hidden_size"] = kwargs.pop("embed_dim")
                
                # Create standard SiglipVisionConfig
                config = SiglipVisionConfig(**kwargs)
                # Force model_type to standard siglip so AutoModel recognizes it
                config.model_type = "siglip_vision_model"
                return config

        # Patch rice_vit and LLaVAOneVision1_5_text into CONFIG_MAPPING
        if hasattr(lo_config.CONFIG_MAPPING, "_extra_content"):
            lo_config.CONFIG_MAPPING._extra_content["rice_vit"] = RiceVitConfigProxy
            lo_config.CONFIG_MAPPING._extra_content["LLaVAOneVision1_5_text"] = Qwen2Config
        
        model = LlavaOVModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=LlavaOnevisionForConditionalGeneration,
            support_models=[llava_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            model_name=cfg.model_name,
            model_args={"trust_remote_code": True}
        )
    elif "qwen" in cfg.model_name.lower():
        qwen_hf_id, model_path = build_model(cfg)
        model = QwenModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModelForCausalLM,
            support_models=[qwen_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name
        )
    elif "llama" in cfg.model_name.lower():  # <-- 新增代码
        # llama_hf_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_path = "/data/share/model_weight/llama/llama-3.1-8b-instruct/"
        llama_hf_id,model_path = build_model(cfg)
        model = LlamaModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModelForCausalLM,
            support_models=[llama_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name
        )
        if model.processor.pad_token is None:
            model.processor.pad_token = model.processor.eos_token

    else:
        raise ValueError(f"Unsupported model: {cfg.model_name}")
    
    device = f"cuda:{getattr(cfg, 'devices', 0)}" if torch.cuda.is_available() else 'cpu'
    model.to(device)


    # Load Dataset
    if cfg.data.name == "mathvista":
        dataset = MathVistaDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "mathvision":
        dataset = MathVisionDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "mmmu":
        dataset = MMMUDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "scienceqa":
        dataset = ScienceQADataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "math_dataset":
        dataset = MATHDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "mmlu":
        dataset = MMLUProDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "gsm8k":
        dataset = GSM8KDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "commonsenseqa":
        dataset = CommonsenseQADataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    elif cfg.data.name == "strategyqa":
        dataset = StrategyQADataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    else:
        # Default/Fallback
        dataset = MATHDataset(cfg.data, model_processor=model.processor, model_name=cfg.model_name)
        dataset_to_process = dataset._support_set
    
    # Use the training set (support_set) for generating self-CoT
    if cfg.max_samples > 0:
        dataset_to_process = dataset_to_process.select(range(min(cfg.max_samples, len(dataset_to_process))))
    
    # Generation arguments
    generation_args = {
        "max_new_tokens": cfg.generation_args.max_new_tokens,
        "temperature": cfg.generation_args.temperature,
        "do_sample": cfg.generation_args.do_sample,
        "top_p": cfg.generation_args.top_p,
        "num_beams": cfg.generation_args.num_beams,
        # "pad_token_id": model.processor.eos_token_id,
    }
    
    # Process samples
    # results = []
    # correct_count = 0
    
    # for i, sample in enumerate(tqdm(dataset_to_process, desc="Generating Self-CoT")):
    #     # --- 1. 获取基本信息 ---
    #     if cfg.data.name == "mathvista":
    #         question = sample.get("query", sample.get("question"))
    #         gt_cot = str(sample.get("answer", ""))
    #         gt_numerical = dataset.extract_answer(gt_cot)
    #         image_val = sample.get("image")
            
    #         # 加载图片
    #         image = None
    #         if isinstance(image_val, Image.Image):
    #             image = image_val
    #         elif isinstance(image_val, str):
    #             full_path = os.path.join(dataset.data_root, image_val)
    #             if os.path.exists(full_path):
    #                 try:
    #                     image = Image.open(full_path).convert("RGB")
    #                 except: image = None
    #     else:
    #         question = sample.get("problem", sample.get("question", ""))
    #         gt_cot = sample.get("solution", sample.get("answer", ""))
    #         gt_numerical = dataset.extract_answer(gt_cot)
    #         image = None

    #     self_cot_response = ""
    #     predicted_answer = ""
    #     is_correct = False

    #     try:
    #         # 执行生成
    #         raw_res = generate_self_cot_for_sample(model, question, generation_args, image=image)
            
    #         # 强制转为字符串 (解决第一个 NoneType 隐患)
    #         self_cot_response = str(raw_res) if raw_res is not None else ""
    #         if self_cot_response == "None": self_cot_response = ""

    #         # 3. 提取与判定 (单独包围，防止提取失败导致数据丢失)
    #         try:
    #             if self_cot_response.strip():
    #                 predicted_answer = extract_answer_from_response(self_cot_response)
    #                 # 只有在 sample 有 choices 时才调用复杂的判定逻辑 (解决第二个 NoneType 隐患)
    #                 if sample and sample.get("choices") is not None:
    #                     is_correct = is_correct_answer(self_cot_response, gt_numerical, sample=sample)
    #                 else:
    #                     is_correct = (str(gt_numerical).lower() in self_cot_response.lower())
                
    #             if is_correct:
    #                 correct_count += 1
    #         except Exception as eval_e:
    #             print(f"Eval warning: {eval_e}")

    #         # 4. 无论如何都保存结果 (确保 Total samples processed 不再缩水)
    #         result = {
    #             "question": question,
    #             "gt_answer": gt_cot,
    #             "gt_numerical": gt_numerical,
    #             "self_cot": self_cot_response,
    #             "predicted_answer": predicted_answer,
    #             "is_correct": is_correct,
    #             "pid": sample.get("pid", "N/A"),
    #             "image": sample.get("image") if isinstance(sample.get("image"), str) else "loaded"
    #         }
    #         results.append(result)
    # Process samples
    results = []
    correct_count = 0

    batch_size = int(cfg.batch_size) if cfg.batch_size is not None else 1
    samples_list = list(dataset_to_process)
    shard_count = int(getattr(cfg, "shard_count", 1) or 1)
    shard_index = int(getattr(cfg, "shard_index", 0) or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"Invalid shard_index={shard_index} for shard_count={shard_count}."
            )
        samples_list = samples_list[shard_index::shard_count]
        print(
            f"Processing shard {shard_index + 1}/{shard_count} with {len(samples_list)} samples."
        )
    else:
        print(f"Processing {len(samples_list)} samples...")

    print(f"Using batch size: {batch_size}")

    for batch_start in tqdm(range(0, len(samples_list), batch_size), desc="Generating Self-CoT"):
        batch_samples = samples_list[batch_start: batch_start + batch_size]

        batch_questions = []
        batch_gt_cot = []
        batch_gt_numerical = []
        batch_images = []
        batch_meta = []

        # 1. 先准备整个 batch 的输入
        for sample in batch_samples:
            if cfg.data.name == "mathvista":
                question = sample.get("query", sample.get("question", ""))
                gt_cot = str(sample.get("answer", ""))
                gt_numerical = dataset.extract_answer(gt_cot)
                image_val = sample.get("image")

                image = None
                if isinstance(image_val, Image.Image):
                    image = image_val
                elif isinstance(image_val, str):
                    full_path = os.path.join(dataset.data_root, image_val)
                    if os.path.exists(full_path):
                        try:
                            image = Image.open(full_path).convert("RGB")
                        except Exception:
                            image = None
            elif cfg.data.name == "mathvision":
                question = dataset.create_query(sample)
                if question.startswith("Question: "):
                    question = question[len("Question: "):]
                gt_cot = str(sample.get("answer", ""))
                gt_numerical = dataset.extract_answer(gt_cot)
                image = dataset._load_image(sample) if use_images else None
            elif cfg.data.name == "mmmu":
                question = dataset.create_query(sample)
                if question.startswith("Question: "):
                    question = question[len("Question: "):]
                gt_cot = str(sample.get("answer", ""))
                gt_numerical = dataset.extract_answer(gt_cot)
                image = dataset._collect_images(sample) if use_images else None
            elif cfg.data.name == "scienceqa":
                question = dataset.create_cot_query(sample)
                gt_cot = str(dataset._gold_letter(sample))
                gt_numerical = gt_cot
                image = dataset._decode_image(sample.get("image")) if use_images else None
            elif cfg.data.name == "commonsenseqa":
                question = sample.get("question", "")
                gt_cot = str(sample.get("answerKey", ""))
                gt_numerical = gt_cot
                options = dataset.preprocess_options(sample["choices"])
                question = question + options
                image = None
            elif cfg.data.name == "strategyqa":
                question = sample.get("question", "")
                gt_cot = str(sample.get("answer", ""))
                gt_numerical = dataset.extract_answer(gt_cot)
                image = None
            else:
                question = sample.get("problem", sample.get("question", ""))
                gt_cot = sample.get("solution", sample.get("answer", ""))
                gt_numerical = dataset.extract_answer(gt_cot)
                image = None

            batch_questions.append(question)
            batch_gt_cot.append(gt_cot)
            batch_gt_numerical.append(gt_numerical)
            batch_images.append(image)
            batch_meta.append(sample)

        # 2. 一次性做 batch generate
        try:
            batch_responses = generate_self_cot_for_batch(
                model=model,
                questions=batch_questions,
                generation_args=generation_args,
                images=batch_images,
                dataset_name=cfg.data.name,
            )
        except Exception as e:
            import traceback
            print(f"🚨 Critical batch error at batch_start={batch_start}: {e}")
            traceback.print_exc()
            batch_responses = [""] * len(batch_samples)

        # 3. 对 batch 中每条样本分别做后处理和保存
        for j, sample in enumerate(batch_meta):
            question = batch_questions[j]
            gt_cot = batch_gt_cot[j]
            gt_numerical = batch_gt_numerical[j]
            self_cot_response = batch_responses[j] if j < len(batch_responses) else ""
            if cfg.data.name in {"scienceqa", "commonsenseqa", "strategyqa"}:
                record_question = sample.get("question", "")
            else:
                record_question = question

            predicted_answer = ""
            is_correct = False

            try:
                if self_cot_response.strip():
                    if cfg.data.name in {"gsm8k", "mathvision", "mmmu", "scienceqa", "commonsenseqa", "strategyqa"}:
                        predicted_answer = dataset.extract_answer(self_cot_response)
                    else:
                        predicted_answer = extract_answer_from_response(self_cot_response)

                    if cfg.data.name == "gsm8k":
                        is_correct = str(predicted_answer).strip() == str(gt_numerical).strip()
                    elif cfg.data.name in {"mathvision", "mmmu", "scienceqa"}:
                        is_correct = dataset._is_correct(sample, predicted_answer, self_cot_response)
                    elif cfg.data.name == "commonsenseqa":
                        is_correct = str(predicted_answer).strip().upper() == str(gt_numerical).strip().upper()
                    elif cfg.data.name == "strategyqa":
                        is_correct = str(predicted_answer).strip().capitalize() == str(gt_numerical).strip().capitalize()
                    elif sample and sample.get("choices") is not None:
                        is_correct = is_correct_answer(self_cot_response, gt_numerical, sample=sample)
                    else:
                        is_correct = (str(gt_numerical).lower() in self_cot_response.lower())

                if is_correct:
                    correct_count += 1

            except Exception as eval_e:
                print(f"Eval warning: {eval_e}")

            result = {
                "question": record_question,
                "gt_answer": gt_cot,
                "gt_numerical": gt_numerical,
                "self_cot": self_cot_response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "pid": sample.get("id", sample.get("pid", "N/A")),
                "image": sample.get("image") if isinstance(sample.get("image"), str) else "loaded"
            }
            if cfg.data.name == "commonsenseqa":
                result["options"] = dataset.preprocess_options(sample["choices"])
            if cfg.data.name == "scienceqa":
                result.update(
                    {
                        "query": question,
                        "hint": sample.get("hint", ""),
                        "subject": sample.get("subject", ""),
                        "topic": sample.get("topic", ""),
                        "choices": sample.get("choices", []),
                        "lecture": sample.get("lecture", ""),
                        "solution": sample.get("solution", ""),
                        "image_signature": dataset._image_signature(sample.get("image")),
                        "answer": sample.get("answer", ""),
                        "gold_letter": dataset._gold_letter(sample),
                    }
                )
            results.append(result)
    
    # Calculate final accuracy
    final_accuracy = correct_count / len(results) * 100
    print(f"\nFinal Results:")
    print(f"Total samples processed: {len(results)}")
    print(f"Correct samples: {correct_count}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    
    # Save results
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    
    # Save all results
    with open(cfg.output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Save only correct samples
    correct_output_path = cfg.output_path.replace('.json', '_correct_only.json')
    with open(correct_output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result['is_correct']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"All results saved to: {cfg.output_path}")
    print(f"Correct-only results saved to: {correct_output_path}")
    print(f"Correct samples available for training: {correct_count}")


if __name__ == "__main__":
    main() 
