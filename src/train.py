import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import paths
from data_module import DataModule
from shift_model import ShiftModel, Strategy
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from termcolor import colored
import hydra
from omegaconf import DictConfig
from internvl_model_wrapper import InternVLModelWrapper
from src.llama_model_wrapper import LlamaModelWrapper
from utils import *
from qwen_model_wrapper import QwenModelWrapper
from qwen_vl_model_wrapper import QwenVLModelWrapper
from recovery_ablation import apply_recovery_vector_initialization, resolve_recovery_loss_weights
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from pytorch_lightning.callbacks import TQDMProgressBar

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    def get_max_epochs():
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 10
        elif "idefics2-8b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 5
        elif "llava" in model_name:
            if num_query_samples <= 500:
                return 10
            return 5
        elif "qwen" in model_name:
            if num_query_samples < 500:
                return 10
            return 5
        elif "llama" in model_name:
            if num_query_samples < 500:
                return 10
            return 5

    def save_when(epoch):
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 200:
                if cfg.data.name == "coco":
                    return epoch >= 5
                return epoch >= 7
            if num_query_samples <= 500:
                return epoch >= 5
            return epoch >= 5
        elif "idefics2-8b":
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 500:
                return epoch >= 5
            return True
        elif "llava" in model_name:
            if num_query_samples <= 1000:
                return epoch >= 5
            return True
        elif "qwen" in model_name:
            if num_query_samples <= 1000:
                return epoch >= 5
            return True
        elif "llama" in model_name:
            if num_query_samples <= 1000:   
                return epoch >= 5
            return True

    max_epochs = cfg.epochs if cfg.epochs else get_max_epochs()
    runname = get_expand_runname(cfg)
    print(colored(f"Training for {runname} on {cfg.model_name}", "light_blue"))

    # Determine the checkpoint path to resume from
    resume_ckpt_path_for_trainer = None

    # Priority 1: Explicitly provided resume_ckpt_path
    if hasattr(cfg, "resume_ckpt_path") and cfg.resume_ckpt_path:
        resume_ckpt_path_for_trainer = cfg.resume_ckpt_path
        if not os.path.exists(resume_ckpt_path_for_trainer):
            raise FileNotFoundError(f"Explicit resume checkpoint not found: {resume_ckpt_path_for_trainer}")
        print(colored(f"Explicitly resuming training from checkpoint: {resume_ckpt_path_for_trainer}", "green"))
    # Priority 2: `cfg.resume` flag (automatically find latest checkpoint for the runname)
    elif cfg.resume:
        save_dir = os.path.join(paths.result_dir, "ckpt", runname)
        if os.path.exists(save_dir):
            exist_ckpt_epochs = [
                int(d.split("-")[-1])
                for d in os.listdir(save_dir)
                if os.path.isdir(os.path.join(save_dir, d)) and "epoch-" in d
            ]
            if exist_ckpt_epochs:
                latest_epoch = max(exist_ckpt_epochs)
                resume_ckpt_path_for_trainer = os.path.join(save_dir, f"epoch-{latest_epoch}")
                print(colored(f"Resuming training from latest checkpoint: {resume_ckpt_path_for_trainer}", "green"))
            else:
                print(f"Resume flag set, but no existing checkpoints found in {save_dir}. Starting new training.")
        else:
            print(f"Resume flag set, but checkpoint directory {save_dir} does not exist. Starting new training.")
            os.makedirs(save_dir, exist_ok=True)
            exist_ckpt_epochs = [
            int(d.split("-")[-1])
            for d in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, d))
        ]
            for i in range(max_epochs):
                if save_when(i) and i not in exist_ckpt_epochs:
                    break
            else:
                print(f"All checkpoints {runname} matched, skip...")
                return
    else:
        print("Starting new training (no resume specified).")

    pl.seed_everything(cfg.data.seed)
    os.makedirs(paths.result_dir, exist_ok=True)

    # wb_logger = WandbLogger(
    #     save_dir=paths.result_dir,
    #     name=runname,
    #     project="Gsm8kCoTVector",
    #     log_model=False,
    # )
    torch.set_float32_matmul_precision("medium")
    cuda_available = torch.cuda.is_available()
    visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"CUDA_VISIBLE_DEVICES={visible_cuda_devices}")
    print(f"torch.cuda.is_available()={cuda_available}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if cuda_available:
        try:
            print(f"torch.cuda.current_device()={torch.cuda.current_device()}")
            print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
        except Exception as exc:
            print(f"Failed to query CUDA device details: {exc}")

    accelerator = "gpu" if cuda_available else "cpu"
    trainer_devices = 1
    print(f"Trainer accelerator={accelerator}, devices={trainer_devices}")

    trainer = pl.Trainer(
        # logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            # RichProgressBar(),
            TQDMProgressBar(), # <-- 替换成 TQDMProgressBar
        ],
        # fast_dev_run=True,
        accelerator=accelerator,
        devices=trainer_devices,
        max_epochs=max_epochs,
        # devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        use_distributed_sampler=False,
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip_val,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        enable_checkpointing=False,
        # resume_from_checkpoint=resume_ckpt_path_for_trainer,
    )

    print(f"Trainer将使用的设备: {trainer.device_ids}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    model_info = build_model(cfg)
    model_name_lower = cfg.model_name.lower()

    if "qwen2.5-vl" in model_name_lower:
        model_id, model_path = model_info
        lmm = QwenVLModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=Qwen2_5_VLForConditionalGeneration,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name,
        )

        if hasattr(lmm.processor, "image_processor"):
            lmm.processor.image_processor.max_pixels = 512 * 28 * 28
            lmm.processor.image_processor.min_pixels = 256 * 28 * 28
    elif "internvl" in model_name_lower:
        model_id, model_path = model_info
        lmm = InternVLModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModel,
            support_models=[model_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"use_fast": False, "padding_side": "left"},
            model_args={
                "output_hidden_states": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_flash_attn": True,
            },
            model_name=cfg.model_name,
        )

    elif "qwen" in model_name_lower:
        qwen_hf_id, model_path = model_info
        lmm = QwenModelWrapper(
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

    elif "llama" in model_name_lower:
        llama_hf_id, model_path = model_info
        lmm = LlamaModelWrapper(
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
    else:
        lmm = model_info

    convert_to_peft(cfg, lmm)

    if getattr(cfg, "enable_gradient_checkpointing", False):
        target_model = getattr(lmm, "model", None)
        checkpoint_model = getattr(target_model, "base_model", target_model)
        if checkpoint_model is not None and hasattr(checkpoint_model, "gradient_checkpointing_enable"):
            checkpoint_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for training.")
        if checkpoint_model is not None and hasattr(checkpoint_model, "enable_input_require_grads"):
            checkpoint_model.enable_input_require_grads()
            print("Enabled input require grads for gradient checkpointing.")
        elif checkpoint_model is not None and hasattr(checkpoint_model, "get_input_embeddings"):
            input_embeddings = checkpoint_model.get_input_embeddings()
            if input_embeddings is not None:
                input_embeddings.register_forward_hook(
                    lambda module, inputs, output: output.requires_grad_(True)
                )
                print("Registered input embedding hook to require grads for gradient checkpointing.")
        if checkpoint_model is not None and hasattr(checkpoint_model, "config"):
            checkpoint_model.config.use_cache = False
            print("Disabled model cache for training.")

    cfg.data.max_seq_len = getattr(cfg.data, "max_seq_len", 2048)

    data_module = DataModule(cfg, lmm)

    only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
    if isinstance(only_shift_at_layer, str):
        try:
            parsed_value = json.loads(only_shift_at_layer)
            if isinstance(parsed_value, (list, int, type(None))):
                only_shift_at_layer = parsed_value
        except json.JSONDecodeError:
            pass

    if isinstance(only_shift_at_layer, int) and only_shift_at_layer < 0:
        start_layer = int(getattr(cfg, "start_layer", 0))
        end_layer = int(getattr(cfg, "end_layer", getattr(cfg, "model_layers", 0)))
        only_shift_at_layer = list(range(start_layer, end_layer)) if end_layer > start_layer else None

    record_only_layers = None
    if getattr(cfg, "record_only_shift_layer_for_alignment", False):
        record_only_layers = only_shift_at_layer
        print(f"Record-only alignment layers enabled: {record_only_layers}")

    record_masked_hidden_states_only = bool(
        getattr(cfg, "record_masked_hidden_states_only", False)
    )
    if record_masked_hidden_states_only:
        print("Record masked hidden states only is enabled.")

    shift_encoder = hydra.utils.instantiate(cfg.encoder.cls, _partial_=True)(
        lmm=lmm,
        only_shift_at_layer=only_shift_at_layer,
        record_only_layers=record_only_layers,
        record_masked_hidden_states_only=record_masked_hidden_states_only,
    )
    shift_encoder.recovery_init_info = apply_recovery_vector_initialization(cfg, shift_encoder)
    loss_mode, effective_ce_loss_weight, effective_align_loss_weight = resolve_recovery_loss_weights(cfg)
    print(
        f"Recovery ablation config: init_mode={cfg.init_mode}, loss_mode={loss_mode}, "
        f"effective_ce_loss_weight={effective_ce_loss_weight}, "
        f"effective_align_loss_weight={effective_align_loss_weight}"
    )

    model = ShiftModel(
        cfg,
        shift_encoder,
        eval(cfg.encoder.model_strategy),
        save_checkpoint_when=save_when,
    )
    trainer.fit(
        model,
        data_module,
    )


if __name__ == "__main__":
    main()
