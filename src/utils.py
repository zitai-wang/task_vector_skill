from contextlib import nullcontext
import json
import os
import torch
import torch.nn as nn
import peft
from peft import PeftModel, LoraConfig, PrefixTuningConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoProcessor
try:
    from internvl_model_wrapper import InternVLModelWrapper
except ModuleNotFoundError:
    InternVLModelWrapper = None
try:
    from llava_ov_model_wrapper import LlavaOVModelWrapper
except ModuleNotFoundError:
    LlavaOVModelWrapper = None
from qwen_vl_model_wrapper import QwenVLModelWrapper
import re
from safetensors import safe_open

import paths
# from testbed.models.llava import LLaVa
# from testbed.models import Idefics, Idefics2

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _normalize_env_key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.upper())


def _model_root_override(model_name: str) -> str | None:
    return os.environ.get(f"COT_MIMIC_MODEL_ROOT_{_normalize_env_key(model_name)}")


def _pick_existing_path(candidates, validator=None):
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        if validator is None or validator(candidate):
            return candidate
    return candidates[0]


class NullPeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        # MimIC-style experiments optimize only the shift encoder, so keep the
        # backbone frozen to avoid unnecessary gradient storage.
        model.requires_grad_(False)
        self.config = model.config

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def disable_adapter(self):
        return nullcontext()

    def save_pretrained(self, save_directory: str, **kwargs):
        pass

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)

    @property
    def device(self):
        if hasattr(self.base_model, "device"):
            return self.base_model.device
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class ModuleDeviceManager:
    def __init__(self, device):
        self.device = device
        self.old_devices = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, old_device in self.old_devices.items():
            module.to(old_device)

    def move_module(self, module):
        self.old_devices[module] = next(module.parameters()).device
        module.to(self.device)


def get_runtime_device():
    device = torch.device("cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        if visible not in (None, "", "-1"):
            try:
                torch.empty(1, device=device)
                gpu_name = torch.cuda.get_device_name(device)
                print(
                    f"Running on GPU via fallback probe: {gpu_name} "
                    f"(CUDA_VISIBLE_DEVICES={visible}, using logical cuda:0)"
                )
                return device
            except Exception as exc:
                print(
                    "CUDA requested but unavailable after fallback probe; "
                    f"falling back to CPU. reason={exc}"
                )
        print("Running on CPU.")
        return torch.device("cpu")

    gpu_name = torch.cuda.get_device_name(device)
    if visible:
        print(
            f"Running on GPU: {gpu_name} "
            f"(CUDA_VISIBLE_DEVICES={visible}, using logical cuda:0)"
        )
    else:
        print(f"Running on GPU: {gpu_name}")
    return device


def convert_to_peft(cfg, lmm):
    peft_config_cls = None
    if cfg.model_name in cfg.peft:
        peft_cfg = OmegaConf.to_container(cfg.peft[cfg.model_name], resolve=True)
        if cfg.peft.name == "lora":
            peft_config_cls = LoraConfig
        elif cfg.peft.name == "prefix-tuning":
            peft_config_cls = PrefixTuningConfig

    if peft_config_cls is None:
        model = NullPeftModel(lmm.model)
    else:
        model = peft.get_peft_model(lmm.model, peft_config_cls(**peft_cfg))
    lmm.model = model


def build_qwen_model_info(cfg):
    def qwen_vl_checkpoint_usable(candidate):
        shard_path = os.path.join(candidate, "model-00002-of-00005.safetensors")
        if not os.path.exists(shard_path):
            return False
        try:
            with safe_open(shard_path, framework="pt"):
                return True
        except Exception as exc:
            print(f"Skipping unusable QwenVL checkpoint at {candidate}: {exc}")
            return False

    # Mapping for Qwen models: model_name -> (hf_id, local_path)
    qwen_models = {
        "qwen2.5-math-7b-instruct": (
            "Qwen/Qwen2.5-Math-7B-Instruct",
            _pick_existing_path([
                _model_root_override("qwen2.5-math-7b-instruct"),
                "/home/share/model_weight/qwen/Qwen2.5-Math-7B-Instruct/",
                "/data/share/model_weight/qwen/Qwen2.5-Math-7B-Instruct/",
            ]),
            # a6000:"/home/share/model_weight/qwen/Qwen2.5-Math-7B-Instruct/"
            # "/home/xiangzhong_guest/ll26/Qwen2.5-Math-7B-Instruct/"
        ),
        "qwen2.5-math-7b": (
            "Qwen/Qwen2.5-Math-7B",
            _pick_existing_path([
                _model_root_override("qwen2.5-math-7b"),
                "/home/share/model_weight/qwen/Qwen2.5-Math-7B/",
                "/data/share/model_weight/qwen/Qwen2.5-Math-7B/",
            ]),
            # "/home/share/model_weight/qwen/Qwen2.5-Math-7B/"
            # "/home/xiangzhong_guest/ll26/Qwen2.5-Math-7B/"
        ),
        # Add other Qwen models here as needed
        "qwen2.5-7b": (
            "Qwen/Qwen2.5-7B",
            _pick_existing_path([
                _model_root_override("qwen2.5-7b"),
                "/home/share/model_weight/qwen/Qwen2.5-7B/",
                "/data/share/model_weight/qwen/Qwen2.5-7B/",
            ]),
        ),
        "qwen2.5-7b-instruct": (
            "Qwen/Qwen2.5-7B-Instruct",
            _pick_existing_path([
                _model_root_override("qwen2.5-7b-instruct"),
                "/home/share/model_weight/qwen/Qwen2.5-7B-Instruct/",
                "/data/share/model_weight/qwen/Qwen2.5-7B-Instruct/",
            ]),
        ),
        "qwen2.5-vl-7b-instruct": (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            _pick_existing_path([
                _model_root_override("qwen2.5-vl-7b-instruct"),
                "/home/dhz/Model/Qwen2.5-VL-7B-Instruct/",
                "/home/lrz/model_weight/Qwen2.5-VL-7B-Instruct/",
                "/home/share/Qwen2.5-VL-7B-Instruct/",
                "/home/share/model_weight/qwen/Qwen2.5-VL-7B-Instruct/",
                "/data/share/Qwen2.5-VL-7B-Instruct/",
            ], validator=qwen_vl_checkpoint_usable),
        ),
    }

    model_name = cfg.model_name.lower()
    if model_name not in qwen_models:
        raise ValueError(f"Unsupported Qwen model: {cfg.model_name}. Please add it to `build_qwen_model_info`.")

    hf_id, local_path = qwen_models[model_name]
    print(f"Returning model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path


def build_llama_model_info(cfg):
    llama_models = {
        "llama-3.1-8b-instruct": (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            _pick_existing_path([
                _model_root_override("llama-3.1-8b-instruct"),
                "/data/share/model_weight/llama/llama-3.1-8b-instruct/",
            ]),
        ),
        "llama-3.1-70b-instruct": (
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            _pick_existing_path([
                _model_root_override("llama-3.1-70b-instruct"),
                "/data/share/model_weight/llama/llama-3.1-70b-instruct/",
            ]),
        ),
        "llama3-8b": (
            "meta-llama/Meta-Llama-3-8B", 
            _pick_existing_path([
                _model_root_override("llama3-8b"),
                _model_root_override("llama-3.1-8b-instruct"),
                "/data/share/model_weight/llama/llama-3.1-8b-instruct/",
            ]), 
        ),
        # Add other LLaMA models here as needed  
    }
    model_name = cfg.model_name.lower()
    if model_name not in llama_models:
        raise ValueError(f"Unsupported LLaMA model: {cfg.model_name}. Please add it to `build_llama_model_info`.")

    hf_id, local_path = llama_models[model_name]
    print(f"Returning LLaMA model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path
    # from transformers import AutoModelForCausalLM
    # return AutoModelForCausalLM.from_pretrained(
    #     local_path,
    #     torch_dtype=eval(cfg.dtype)
    # )


def build_internlm_model_info(cfg):
    internlm_models = {
        "internlm2_5-7b-chat": (
            "internlm/internlm2_5-7b-chat",
            _pick_existing_path([
                _model_root_override("internlm2_5-7b-chat"),
                "/data/share/internlm2_5-7b-chat/",
            ]),
        ),
    }
    model_name = cfg.model_name.lower()
    if model_name not in internlm_models:
        raise ValueError(
            f"Unsupported InternLM model: {cfg.model_name}. Please add it to `build_internlm_model_info`."
        )

    hf_id, local_path = internlm_models[model_name]
    print(f"Returning InternLM model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path


def build_internvl_model_info(cfg):
    internvl_models = {
        "internvl2_5-8b": (
            "OpenGVLab/InternVL2_5-8B",
            _pick_existing_path([
                _model_root_override("internvl2_5-8b"),
                "/data/share/InternVL2_5-8B",
            ]),
        ),
    }
    model_name = cfg.model_name.lower()
    if model_name not in internvl_models:
        raise ValueError(
            f"Unsupported InternVL model: {cfg.model_name}. Please add it to `build_internvl_model_info`."
        )

    hf_id, local_path = internvl_models[model_name]
    print(f"Returning InternVL model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path

def build_llava_model_info(cfg):
    llava_models={
        "llava-onevision-1.5-8b": (
            "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
            _pick_existing_path([
                _model_root_override("llava-onevision-1.5-8b"),
                "/data/share/LLaVA-OneVision-1.5-8B-Instruct/",
            ]),
        ),
    }
    model_name = cfg.model_name.lower()
    if model_name not in llava_models:
        raise ValueError(f"Unsupported LLaMA model: {cfg.model_name}. Please add it to `build_llama_model_info`.")
    
    hf_id, local_path = llava_models[model_name]
    print(f"Returning model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path


def build_idefics3_model_info(cfg):
    idefics3_models = {
        "idefics3-8b-llama3": (
            "HuggingFaceM4/Idefics3-8B-Llama3",
            _pick_existing_path([
                _model_root_override("idefics3-8b-llama3"),
                "/data/share/Idefics3-8B-Llama3/",
            ]),
        ),
    }
    model_name = cfg.model_name.lower()
    if model_name not in idefics3_models:
        raise ValueError(
            f"Unsupported Idefics3 model: {cfg.model_name}. Please add it to `build_idefics3_model_info`."
        )

    hf_id, local_path = idefics3_models[model_name]
    print(f"Returning Idefics3 model info: path={local_path}, HF_ID={hf_id}")
    return hf_id, local_path


def build_model(cfg):
    """
    Builds the main model based on the configuration.
    """
    model_name = cfg.model_name

    if "idefics" in model_name:
        if model_name == "idefics-9b":
            lmm = Idefics(
                paths.idefics_9b_path,
                torch_dtype=eval(cfg.dtype),
            )
        elif model_name == "idefics2-8b-base":
            processor_args = {
                "do_image_splitting": False,
            }
            if "seed" in cfg.data.name or "mme" in cfg.data.name:
                # seed bench cannot even run 1 shot with the default setting
                processor_args["largest_edges"] = 448
                processor_args["shortest_edges"] = 378

            lmm = Idefics2(
                paths.idefics2_8b_base_path,
                torch_dtype=eval(cfg.dtype),
                processor_args=processor_args,
            )
        else:
            raise ValueError(f"Unsupport model {model_name}")
        return lmm
    elif "llava-onevision" in model_name or "llava-ov" in model_name:
        if LlavaOVModelWrapper is None:
            raise ModuleNotFoundError(
                "llava_ov_model_wrapper is required only for LLaVA-OneVision runs. "
                "Non-LLaVA runs can continue without it."
            )
        hf_id, model_path = build_llava_model_info(cfg)
        print(f"Instantiating LlavaOVModelWrapper for {model_name}...")
        lmm = LlavaOVModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=AutoModelForCausalLM,
            support_models=[hf_id],
            torch_dtype=eval(cfg.dtype),
            model_name=model_name,
        )
        return lmm
    elif "idefics3" in model_name:
        try:
            from idefics3_model_wrapper import Idefics3ModelWrapper
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "idefics3_model_wrapper is required only for Idefics3 runs. "
                "Your current QwenVL runs can work without it, but an Idefics3 "
                "experiment needs that module restored."
            ) from exc
        hf_id, model_path = build_idefics3_model_info(cfg)
        print(f"Instantiating Idefics3ModelWrapper for {model_name}...")
        lmm = Idefics3ModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            support_models=[hf_id],
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_name=model_name,
        )
        return lmm
    elif "qwen_vl" in model_name:
        hf_id, model_path = build_qwen_model_info(cfg)
        print(f"Instantiating QwenVLModelWrapper for {model_name}...")
        lmm = QwenVLModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor, # VL 模型通常用 Processor
            model_class=AutoModelForCausalLM, 
            support_models=[hf_id],
            torch_dtype=eval(cfg.dtype),
            model_name=model_name,
            # device_map="auto" # 如果需要自动多卡
        )
        return lmm
    elif "qwen2.5" in model_name:
        qwen_hf_id, model_path = build_qwen_model_info(cfg)
        return qwen_hf_id, model_path  # Return the info, not the loaded model
    elif "llama" in model_name:
        llama_hf_id, model_path = build_llama_model_info(cfg)
        return llama_hf_id, model_path
    elif "internlm" in model_name:
        internlm_hf_id, model_path = build_internlm_model_info(cfg)
        return internlm_hf_id, model_path
    elif "internvl" in model_name:
        internvl_hf_id, model_path = build_internvl_model_info(cfg)
        return internvl_hf_id, model_path
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(
        #     "/data/share/model_weight/llama/llama-3.1-8b-instruct/",
        #     torch_dtype=torch.float16,
        #     device_map="cuda:0"
        # )
        # return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def save_pretrained(save_directory, lmm, encoder):
    os.makedirs(save_directory, exist_ok=True),
    encoder_sd = {
        k: v for k, v in encoder.state_dict().items() if not k.startswith("lmm")
    }
    torch.save(encoder_sd, os.path.join(save_directory, "encoder.pth"))
    lmm.model.save_pretrained(save_directory)


def load_from_pretrained(save_directory, lmm, encoder):
    sd = torch.load(os.path.join(save_directory, "encoder.pth"), weights_only=True)
    if sd:
        missing_keys, unexpected_keys = encoder.load_state_dict(sd, strict=False)

        # keys started with "lmm" is not related to shift encoder
        missing_keys = [k for k in missing_keys if not k.startswith("lmm")]
        if missing_keys:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if os.path.exists(os.path.join(save_directory, "adapter_config.json")):
        lmm.model = PeftModel.from_pretrained(encoder.lmm.model, save_directory)


def parse_only_shift_at_layer(value):
    if not isinstance(value, str):
        return value

    try:
        parsed_value = json.loads(value)
        if isinstance(parsed_value, (list, int, type(None))):
            return parsed_value
    except json.JSONDecodeError:
        pass

    stripped = value.strip()
    if stripped.lower() == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return value


# update execute_eval in eval_old.py if runname format is changed
def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train, eval or analyze config.
    """
    if hasattr(cfg, "runname"):
        # training mode
        if cfg.data.num_shot == 0:
            # peft
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        # runname-model-dataset-training_samples-num_shot
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

    if hasattr(cfg, "record_dir"):
        # analyze mode
        # record dir format: path/to/{expand-runname}
        return os.path.basename(cfg.record_dir)

    if getattr(cfg, "ckpt_path", None):
        # eval from a specific checkpoint
        # checkpoint dir format: path/to/{expand-runname}/epoch-{epoch}
        return os.path.basename(os.path.dirname(cfg.ckpt_path))
    else:
        # eval from ICL
        # icl-model-dataset-[num_shot]shot-[cot/direct]
        cot_suffix = "cot" if cfg.use_cot else "direct"
        return f"icl-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_shot}shot-{cot_suffix}"
