import enum
from functools import reduce
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
from datetime import datetime
from typing import Optional

from omegaconf import OmegaConf

sys.path.insert(0, "..")
import paths
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from dataset_utils import dataset_mapping
from recovery_ablation import (
    get_trainable_vector_stats,
    resolve_recovery_loss_weights,
)
from utils import save_pretrained, get_expand_runname


class Strategy(enum.IntFlag):
    LAYER_WISE_MSE = 2
    LAYER_WISE_COS_SIM = 64  # equivalent to normalized L2 distance
    LOGITS_KL_DIV = 4
    LM_LOSS = 8

    def has_layer_wise(self):
        try:
            self.layer_wise_strategy()
            return True
        except ValueError:
            return False

    def validate(self):
        layer_wise_loss = [
            Strategy.LAYER_WISE_MSE,
            Strategy.LAYER_WISE_COS_SIM,
        ]

        if bin(self & reduce(lambda x, y: x | y, layer_wise_loss)).count("1") > 1:
            raise ValueError(
                f"{[e.name for e in layer_wise_loss]} are mutually exclusive."
            )

    def layer_wise_strategy(self):
        if Strategy.LAYER_WISE_MSE in self:
            return "mse_loss"
        elif Strategy.LAYER_WISE_COS_SIM in self:
            return "cos_sim"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        shift_encoder,
        strategy: Strategy,
        save_checkpoint_when=None,  # should be a method f(epoch), save the last ckpt by default
    ) -> None:
        super().__init__()
        self.lmm = shift_encoder.lmm
        self.cfg = cfg
        self.shift_encoder = shift_encoder
        strategy.validate()
        self.strategy = strategy
        self.save_checkpoint_when = (
            save_checkpoint_when
            if save_checkpoint_when is not None
            else lambda epoch: epoch == self.trainer.max_epochs - 1
        )
        self.save_dir = os.path.join(paths.result_dir, "ckpt", get_expand_runname(cfg))
        self.record_dir = os.path.join(
            paths.result_dir,
            "record",
            str(getattr(getattr(self.cfg, "validation_eval", {}), "record_subdir", "learnable")),
            str(getattr(self.cfg.data, "name", "unknown")),
            get_expand_runname(cfg),
        )
        self.metrics_dir = os.path.join(self.save_dir, "recovery_metrics")
        self.best_checkpoint_path = None
        self.final_checkpoint_path = None
        self.best_accuracy = float("-inf")
        self.best_epoch = None
        self.final_accuracy = None
        self.final_epoch = None
        self.initial_direct_accuracy = None
        self.initial_direct_record_path = None
        self.epoch_train_sums = {}
        self.epoch_train_count = 0
        self.epoch_eval_history = []
        self._last_seen_global_step = -1
        self._last_late_checkpoint_step = -1
        self.loss_mode, self.effective_ce_loss_weight, self.effective_align_loss_weight = (
            resolve_recovery_loss_weights(self.cfg)
        )
        self.init_info = getattr(self.shift_encoder, "recovery_init_info", {})

    def _ensure_internvl_text_only_inputs(self, model_inputs: dict) -> dict:
        model_name = str(getattr(self.cfg, "model_name", "")).lower()
        if "internvl" not in model_name or "pixel_values" in model_inputs:
            return model_inputs

        input_ids = model_inputs["input_ids"]
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        input_size = int(getattr(self.lmm, "input_size", 448))
        vision_dtype = getattr(getattr(self.lmm, "model", None), "dtype", torch.float32)

        patched = dict(model_inputs)
        # InternVL forward always requires pixel_values/image_flags, even when the
        # prompt contains no <image> tokens. Provide a zero-image placeholder and
        # mark every sample as text-only with image_flags=0.
        patched["pixel_values"] = torch.zeros(
            (batch_size, 3, input_size, input_size),
            device=device,
            dtype=vision_dtype,
        )
        patched["image_flags"] = torch.zeros(
            (batch_size, 1),
            device=device,
            dtype=torch.long,
        )
        return patched

    @staticmethod
    def _scalar_to_float(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().item())
        return float(value)

    def _append_jsonl(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _reset_epoch_train_metrics(self) -> None:
        self.epoch_train_sums = {
            "ce_loss": 0.0,
            "align_loss": 0.0,
            "total_loss": 0.0,
            "alignment_gap": 0.0,
        }
        self.epoch_train_count = 0

    def _update_epoch_train_metrics(self, loss_dict: dict) -> None:
        for key in self.epoch_train_sums:
            self.epoch_train_sums[key] += self._scalar_to_float(loss_dict[key])
        self.epoch_train_count += 1

    def _current_vector_stats(self) -> dict:
        return get_trainable_vector_stats(self.shift_encoder)

    def _late_checkpoint_cfg(self):
        return getattr(self.cfg, "late_checkpoint", None)

    def _maybe_save_late_checkpoint(self):
        late_cfg = self._late_checkpoint_cfg()
        if not late_cfg or not bool(getattr(late_cfg, "enabled", False)):
            return

        current_step = int(self.global_step)
        if current_step <= 0 or current_step == self._last_late_checkpoint_step:
            return

        start_epoch = int(getattr(late_cfg, "start_epoch", 0))
        if int(self.current_epoch) < start_epoch:
            return

        every_n_optimizer_steps = int(getattr(late_cfg, "every_n_optimizer_steps", 0))
        if every_n_optimizer_steps <= 0 or current_step % every_n_optimizer_steps != 0:
            return

        ckpt_dir = os.path.join(
            self.save_dir,
            f"epoch-{int(self.current_epoch)}-step-{current_step}",
        )
        save_pretrained(ckpt_dir, self.lmm, self.shift_encoder)
        self._last_late_checkpoint_step = current_step
        print(
            "[LateCheckpoint] "
            f"epoch={int(self.current_epoch)} "
            f"global_step={current_step} "
            f"path={ckpt_dir}"
        )

    def _step_metrics_path(self) -> str:
        return os.path.join(self.metrics_dir, "train_step_metrics.jsonl")

    def _epoch_metrics_path(self) -> str:
        return os.path.join(self.metrics_dir, "train_epoch_metrics.jsonl")

    def _eval_metrics_path(self) -> str:
        return os.path.join(self.metrics_dir, "val_epoch_metrics.jsonl")

    def _summary_path(self) -> str:
        return os.path.join(self.save_dir, "recovery_ablation_summary.json")

    def _record_eval_path(self, epoch: int) -> str:
        return os.path.join(self.record_dir, f"epoch-{epoch}.json")

    def _initial_eval_path(self) -> str:
        return os.path.join(self.record_dir, "initial_direct_eval.json")

    def _write_epoch_train_metrics(self) -> None:
        if self.epoch_train_count == 0 or self.global_rank != 0:
            return

        averages = {
            key: value / self.epoch_train_count for key, value in self.epoch_train_sums.items()
        }
        vector_stats = self._current_vector_stats()
        payload = {
            "epoch": int(self.current_epoch),
            "train_batches": int(self.epoch_train_count),
            **averages,
        }
        payload.update(vector_stats)
        self._append_jsonl(self._epoch_metrics_path(), payload)
        print(
            "[EpochTrain] "
            f"epoch={payload['epoch']} "
            f"batches={payload['train_batches']} "
            f"ce_loss={payload['ce_loss']:.6f} "
            f"align_loss={payload['align_loss']:.6f} "
            f"total_loss={payload['total_loss']:.6f} "
            f"alignment_gap={payload['alignment_gap']:.6f} "
            f"vector_norm={vector_stats['norm']:.6f}"
        )

    def _save_eval_record(self, epoch: int, eval_payload: dict) -> None:
        os.makedirs(self.record_dir, exist_ok=True)
        with open(self._record_eval_path(epoch), "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, ensure_ascii=False)

    def _build_validation_dataset(self):
        eval_data_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.data, resolve=True))
        validation_cfg = getattr(self.cfg, "validation_eval", None)
        if validation_cfg is not None:
            if getattr(validation_cfg, "query_split", None):
                eval_data_cfg.query_split = validation_cfg.query_split
            if getattr(validation_cfg, "support_split", None):
                eval_data_cfg.support_split = validation_cfg.support_split

        dataset = dataset_mapping[self.cfg.data.name](
            eval_data_cfg,
            model_processor=self.lmm.processor,
            model_name=self.cfg.model_name,
        )

        max_eval_samples = getattr(validation_cfg, "max_eval_samples", None) if validation_cfg else None
        if max_eval_samples is not None and hasattr(dataset, "_query_set") and max_eval_samples > 0:
            capped = min(int(max_eval_samples), len(dataset._query_set))
            dataset._query_set = dataset._query_set.select(list(range(capped)))
        return dataset, eval_data_cfg

    def _run_validation_eval(self) -> Optional[dict]:
        validation_cfg = getattr(self.cfg, "validation_eval", None)
        if not validation_cfg or not bool(getattr(validation_cfg, "enabled", False)):
            return None

        dataset, eval_data_cfg = self._build_validation_dataset()
        eval_cfg = OmegaConf.create(
            {
                "model_name": self.cfg.model_name,
                "batch_size": int(getattr(validation_cfg, "batch_size", 1)),
                "eval_mode": str(getattr(validation_cfg, "eval_mode", "EVAL_WITH_COT_VECTOR_DIRECT_Q")),
                "generation_args": OmegaConf.to_container(
                    validation_cfg.generation_args,
                    resolve=True,
                ),
                "data": OmegaConf.to_container(eval_data_cfg, resolve=True),
            }
        )

        hooks = self.shift_encoder.register_shift_hooks()
        self.shift_encoder.eval()
        self.lmm.eval()
        try:
            records, eval_result = dataset.eval(eval_cfg, self.lmm)
        finally:
            self.shift_encoder.remove_hooks(hooks)
            self.shift_encoder.train()
            self.lmm.model.train()

        vector_stats = self._current_vector_stats()
        align_gap_values = [entry["alignment_gap"] for entry in self.epoch_eval_history]
        payload = {
            "epoch": int(self.current_epoch),
            "eval_args": OmegaConf.to_container(eval_cfg, resolve=True),
            "eval_result": eval_result,
            "records": records,
            "alignment_gap": self.epoch_train_sums["alignment_gap"] / max(self.epoch_train_count, 1),
            "average_alignment_gap_so_far": (
                sum(align_gap_values) / len(align_gap_values) if align_gap_values else 0.0
            ),
            "trainable_vector_stats": vector_stats,
            "init_info": self.init_info,
        }
        return payload

    def _write_summary(self) -> None:
        if self.global_rank != 0:
            return

        vector_stats = self._current_vector_stats()
        avg_alignment_gap = 0.0
        if self.epoch_eval_history:
            avg_alignment_gap = sum(x["alignment_gap"] for x in self.epoch_eval_history) / len(
                self.epoch_eval_history
            )

        summary = {
            "benchmark": str(getattr(self.cfg, "benchmark_name", self.cfg.data.name)),
            "method_name": str(
                getattr(self.cfg, "method_name", "trainable_multimodal_extracted_vector_recovery")
            ),
            "recovery_vector_semantics": str(
                getattr(self.cfg, "recovery_vector_semantics", "mimic")
            ),
            "init_mode": str(getattr(self.cfg, "init_mode", "random")),
            "loss_mode": str(self.loss_mode),
            "final_accuracy": self.final_accuracy,
            "initial_direct_accuracy": self.initial_direct_accuracy,
            "initial_direct_record_path": self.initial_direct_record_path,
            "best_epoch": self.best_epoch,
            "best_checkpoint_path": self.best_checkpoint_path,
            "final_epoch": self.final_epoch,
            "final_checkpoint_path": self.final_checkpoint_path,
            "average_alignment_gap": avg_alignment_gap,
            "trainable_vector_norm": vector_stats["norm"],
            "trainable_vector_stats": vector_stats,
            "train_total_epochs": int(self.trainer.max_epochs),
            "learning_rate": float(getattr(self.cfg, "lr", 0.0)),
            "extract_source_model_name": self.init_info.get("source_model_name"),
            "extract_dataset_name": self.init_info.get("source_dataset_name"),
            "extract_source_path": self.init_info.get("source_path"),
            "ce_loss_weight": self.effective_ce_loss_weight,
            "align_loss_weight": self.effective_align_loss_weight,
            "save_dir": self.save_dir,
            "record_dir": self.record_dir,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self._summary_path(), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _should_skip_oom(self, exc) -> bool:
        dataset_name = str(getattr(self.cfg.data, "name", "")).lower()
        return (
            bool(getattr(self.cfg, "skip_oom_batches", False))
            and dataset_name == "mmmu"
            and "out of memory" in str(exc).lower()
        )

    def _handle_oom_skip(self, stage: str, batch_idx=None) -> None:
        # Clear any partially accumulated gradients before continuing.
        self.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_msg = "" if batch_idx is None else f", batch_idx={batch_idx}"
        print(
            f"Warning: skipped OOM during {stage} at epoch={self.current_epoch}"
            f"{batch_msg}. Continuing training."
        )
        self.log(
            "oom_skip",
            1.0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_start(self):
        if self.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.metrics_dir, exist_ok=True)
        self._reset_epoch_train_metrics()
        if self.global_rank == 0:
            initial_eval = self._run_validation_eval()
            if initial_eval is not None:
                initial_path = self._initial_eval_path()
                os.makedirs(self.record_dir, exist_ok=True)
                with open(initial_path, "w", encoding="utf-8") as f:
                    json.dump(initial_eval, f, ensure_ascii=False)
                self.initial_direct_accuracy = float(initial_eval["eval_result"].get("accuracy", 0.0))
                self.initial_direct_record_path = initial_path
                print(
                    f"[InitialEval] accuracy={self.initial_direct_accuracy:.4f} "
                    f"record_path={self.initial_direct_record_path}"
                )

    def on_train_epoch_start(self):
        self._reset_epoch_train_metrics()

    def generate_label_mask(self, inputs, num_separator, keep_bos=False):
        """
        Generates label mask which masks tokens before num_separator pad_tokens from given inputs.
        """
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        pad_mask = input_ids == self.lmm.processor.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lmm.processor.padding_side == "left":
            bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if self.lmm.processor.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]

            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=self.device), bos_position] = (
                True
            )

        return label_mask

    def remove_hooks(self, hooks):
        # remove all hooks
        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, query_label_mask):
        """
        Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
        and convert to batch_size * [num_layer, query_part_len, d_model].
        """
        hidden_states_dict = {}

        for name, attr in vars(self.shift_encoder).items():
            if name.endswith("_hidden_states"):
                if isinstance(attr, dict):
                    ordered_hidden_states = []
                    missing_layers = []
                    for layer_idx in sorted(attr):
                        layer_hidden_states = attr[layer_idx]
                        if layer_hidden_states is None:
                            missing_layers.append(layer_idx)
                        else:
                            ordered_hidden_states.append(layer_hidden_states)
                    if missing_layers:
                        raise RuntimeError(
                            f"Missing recorded hidden states for layers: {missing_layers}"
                        )
                    if not ordered_hidden_states:
                        raise RuntimeError(
                            f"No hidden states were recorded for {name}."
                        )
                    layer_values = ordered_hidden_states
                else:
                    missing_layers = [
                        layer_idx
                        for layer_idx, layer_hidden_states in enumerate(attr)
                        if not (torch.is_tensor(layer_hidden_states) or isinstance(layer_hidden_states, list))
                    ]
                    if missing_layers:
                        raise RuntimeError(
                            f"Missing recorded hidden states for layers: {missing_layers}"
                        )
                    layer_values = attr

                if layer_values and isinstance(layer_values[0], list):
                    batch_size = len(layer_values[0])
                    num_layer = len(layer_values)
                    extracted_hss = []
                    for batch_idx in range(batch_size):
                        per_layer_hidden_states = [
                            layer_hidden_states[batch_idx] for layer_hidden_states in layer_values
                        ]
                        extracted_hss.append(torch.stack(per_layer_hidden_states, dim=0))
                else:
                    # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                    hidden_states_all_layers = torch.stack(layer_values).transpose(0, 1)
                    batch_size, num_layer, seq_len, d_model = hidden_states_all_layers.shape

                    extracted_hss = []
                    for i in range(batch_size):
                        # hs_for_batch_element: (NumLayers, SeqLen, Dmodel) for this batch element
                        hs_for_batch_element = hidden_states_all_layers[i]
                        mask_for_batch_element = query_label_mask[i] # (SeqLen,)

                        # Get the indices of the masked elements and select them
                        extracted_hs_i = hs_for_batch_element.masked_select(mask_for_batch_element[None, :, None]).view(num_layer, -1, d_model)
                        extracted_hss.append(extracted_hs_i)

                hidden_states_dict[name] = extracted_hss # This is now a list of (num_layer, query_part_len, d_model)
                                                                # which can be stacked in calculate_layer_wise_loss
        if not hidden_states_dict:
            raise RuntimeError(
                "Layer wise loss requires to record hidden states, but no any *_hidden_states in shift encoder."
            )

        return hidden_states_dict

    def calculate_layer_wise_loss(self, shift_hidden_states, prefix_hidden_states):
        alignment_length_mismatch_mode = str(
            getattr(self.cfg, "alignment_length_mismatch_mode", "mean_pool")
        ).lower()

        if Strategy.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(
                input.float(),  # Cast to float32
                target.float(), # Cast to float32
                reduction="mean",
            )
        elif Strategy.LAYER_WISE_COS_SIM in self.strategy:
            loss_fn = lambda input, target: 1 - torch.mean(
                F.cosine_similarity(
                    input.float(), # Cast to float32
                    target.float(), # Cast to float32
                    dim=-1,
                ),
                dim=1,
            )

        layer_loss = dict()
        for (shift_hs_varname, shift_hs_list), (_, prefix_hs_list) in zip(
            shift_hidden_states.items(), prefix_hidden_states.items()
        ):
            per_sample_losses = []
            for shift_hs, prefix_hs in zip(shift_hs_list, prefix_hs_list):
                if shift_hs.shape != prefix_hs.shape:
                    if (
                        alignment_length_mismatch_mode == "mean_pool"
                        and shift_hs.dim() == 3
                        and prefix_hs.dim() == 3
                        and shift_hs.shape[0] == prefix_hs.shape[0]
                        and shift_hs.shape[-1] == prefix_hs.shape[-1]
                    ):
                        shift_hs = shift_hs.mean(dim=1, keepdim=True)
                        prefix_hs = prefix_hs.mean(dim=1, keepdim=True)
                    elif (
                        alignment_length_mismatch_mode == "truncate_min"
                        and shift_hs.dim() == 3
                        and prefix_hs.dim() == 3
                        and shift_hs.shape[0] == prefix_hs.shape[0]
                        and shift_hs.shape[-1] == prefix_hs.shape[-1]
                    ):
                        min_len = min(shift_hs.shape[1], prefix_hs.shape[1])
                        shift_hs = shift_hs[:, :min_len, :]
                        prefix_hs = prefix_hs[:, :min_len, :]
                    else:
                        raise RuntimeError(
                            "Alignment hidden state shape mismatch: "
                            f"student={tuple(shift_hs.shape)} teacher={tuple(prefix_hs.shape)} "
                            f"(mode={alignment_length_mismatch_mode})"
                        )
                per_sample_losses.append(loss_fn(shift_hs, prefix_hs))

            layer_loss[
                shift_hs_varname.replace(
                    "hidden_states", self.strategy.layer_wise_strategy()
                )
            ] = torch.mean(torch.stack(per_sample_losses))
        return layer_loss
    
    def calculate_logits_kl_loss(
        self, shift_logits, prefix_logits, query_label_inputs, prefix_label_mask
    ):
        # Ensure the masks are boolean and on the same device as logits
        query_label_inputs = query_label_inputs.to(shift_logits.device)
        prefix_label_mask = prefix_label_mask.to(prefix_logits.device)

        # This operation creates large intermediate tensors if the masks are not sparse
        selected_shift_logits = shift_logits[query_label_inputs]
        selected_prefix_logits = prefix_logits[prefix_label_mask]

        logits_kl_loss = F.kl_div(
            selected_shift_logits.log_softmax(dim=-1),
            selected_prefix_logits.softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        return {"logits_kl_loss": logits_kl_loss}

    def forward(
        self,
        prefix_inputs,
        teacher_answer_mask,
        student_inputs,
        student_labels,
        teacher_cot_mask=None,
        student_answer_mask=None,
        images=None,
        **_unused_batch_kwargs,
    ):
        device = prefix_inputs["input_ids"].device
        zero = torch.zeros((), device=device)
        loss_dict = {
            "loss": zero.clone(),
            "ce_loss": zero.clone(),
            "align_loss": zero.clone(),
            "alignment_gap": zero.clone(),
        }
        need_kl_loss = Strategy.LOGITS_KL_DIV in self.strategy
        is_qwen25_vl = "qwen2.5-vl" in self.cfg.model_name.lower()
        use_low_memory_forward = bool(getattr(self.cfg, "enable_low_memory_forward", False))
        record_masked_hidden_states_only = bool(
            getattr(self.cfg, "record_masked_hidden_states_only", False)
        )

        alignment_teacher_mask = str(
            getattr(self.cfg, "alignment_teacher_mask", "cot_if_available")
        ).lower()
        if (
            alignment_teacher_mask in {"cot", "cot_if_available", "auto"}
            and teacher_cot_mask is not None
        ):
            teacher_hidden_state_mask = teacher_cot_mask.bool()
        else:
            teacher_hidden_state_mask = teacher_answer_mask.bool()

        if student_answer_mask is not None:
            student_prompt_hidden_state_mask = (
                student_answer_mask.bool() & student_inputs["attention_mask"].bool()
            )
        else:
            student_prompt_hidden_state_mask = (
                (student_labels != -100) & student_inputs["attention_mask"].bool()
            )

        eos_token_id = (
            self.lmm.processor.tokenizer.eos_token_id
            if hasattr(self.lmm.processor, "tokenizer")
            else getattr(self.lmm.processor, "eos_token_id", None)
        )
        if eos_token_id is not None:
            student_prompt_hidden_state_mask &= student_inputs["input_ids"].ne(eos_token_id)

        # Step 1: Teacher model forward pass (no_grad, disable_adapter)
        # Teacher input: Q + Full CoT + Answer
        teacher_hooks = self.shift_encoder.register_record_hooks()
        with torch.no_grad(), self.lmm.model.disable_adapter():
            teacher_forward_kwargs = self._ensure_internvl_text_only_inputs(dict(prefix_inputs))
            if use_low_memory_forward:
                teacher_forward_kwargs["output_hidden_states"] = False
                teacher_forward_kwargs["use_cache"] = False
            if record_masked_hidden_states_only:
                self.shift_encoder.current_recording_mask = teacher_hidden_state_mask
            teacher_outputs = self.lmm.model(**teacher_forward_kwargs)
            prefix_logits = teacher_outputs["logits"] if need_kl_loss else None

            prefix_hidden_states = (
                self.get_hidden_states(teacher_hidden_state_mask)
                if self.strategy.has_layer_wise()
                else None
            )
            # Use the provided teacher_answer_mask directly for prefix_label_mask
            prefix_label_mask = teacher_answer_mask.bool()
            if record_masked_hidden_states_only:
                self.shift_encoder.current_recording_mask = None
        self.remove_hooks(teacher_hooks)
        self.shift_encoder.reset_hidden_states()

        # Step 2: Student model forward pass (with shift hooks enabled)
        # Student input: 1-shot example + Q + GT_Final_Answer (for labels)
        student_hooks = self.shift_encoder.register_shift_hooks()
        # Register record hooks after shift hooks so base/ffn alignment observes
        # the post-injection hidden states instead of the pre-shift module output.
        student_hooks.update(self.shift_encoder.register_record_hooks())

        compute_lm_loss = (
            Strategy.LM_LOSS in self.strategy
            and self.effective_ce_loss_weight != 0.0
        )

        student_forward_kwargs = self._ensure_internvl_text_only_inputs(dict(student_inputs))
        if use_low_memory_forward:
            student_forward_kwargs["output_hidden_states"] = False
            student_forward_kwargs["use_cache"] = False
        if compute_lm_loss:
            student_forward_kwargs["labels"] = student_labels
        if record_masked_hidden_states_only:
            self.shift_encoder.current_recording_mask = student_prompt_hidden_state_mask
        student_outputs = self.lmm.model(**student_forward_kwargs)
        student_logits = student_outputs["logits"] if need_kl_loss else None
        if record_masked_hidden_states_only:
            self.shift_encoder.current_recording_mask = None

        # Calculate student's LM loss (cross-entropy on final answer part)
        if compute_lm_loss:
            if self.cfg.dft_loss:
                logits = student_outputs["logits"]
                labels = student_labels
                # DFT Loss核心实现
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                valid_mask = shift_labels.ne(-100)

                if valid_mask.any():
                    valid_logits = shift_logits[valid_mask]
                    valid_labels = shift_labels[valid_mask]
                    ce_loss = torch.nn.functional.cross_entropy(
                        valid_logits,
                        valid_labels,
                        reduction="none",
                    )
                    probs = torch.softmax(valid_logits, dim=-1)
                    target_probs = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
                    dft_loss = ce_loss * target_probs.detach()
                    loss_dict["ce_loss"] = dft_loss.mean()
                else:
                    loss_dict["ce_loss"] = shift_logits.new_zeros(())
            else:
                loss_dict["ce_loss"] = student_outputs["loss"]

            loss_dict["loss"] += self.effective_ce_loss_weight * loss_dict["ce_loss"]

        shift_hidden_states = (
            self.get_hidden_states(student_prompt_hidden_state_mask.bool())
            if self.strategy.has_layer_wise()
            else None
        )

        self.remove_hooks(student_hooks)

        # Step 3: Calculate layer-wise alignment loss
        if self.strategy.has_layer_wise():
            if prefix_hidden_states is None or shift_hidden_states is None:
                raise RuntimeError("Hidden states not available for layer-wise loss.")

            layer_loss = self.calculate_layer_wise_loss(
                shift_hidden_states, prefix_hidden_states
            )
            loss_dict.update(layer_loss)
            loss_dict["align_loss"] = sum(layer_loss.values())
            loss_dict["alignment_gap"] = torch.mean(torch.stack(list(layer_loss.values())))
            loss_dict["loss"] += self.effective_align_loss_weight * loss_dict["align_loss"]

        # step 4. calculate the last logits kl div
        if Strategy.LOGITS_KL_DIV in self.strategy:
            logits_kl_loss = self.calculate_logits_kl_loss(
                student_logits,
                prefix_logits,
                student_prompt_hidden_state_mask, # Use student's answer mask
                prefix_label_mask,
            )
            loss_dict.update(logits_kl_loss)
            loss_dict["loss"] += self.effective_align_loss_weight * sum(
                logits_kl_loss.values()
            )
        # The original KL_DIV loss is not needed for the new approach, as standard CE and layer alignment are used.
        # self.log('train_loss', loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)
        # self.log('train_align_loss', loss_dict['ffn_cos_sim'], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)
        # self.log('train_ce_loss', loss_dict["ce_loss"], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)

        # print(loss_dict["loss"])
        loss_dict["total_loss"] = loss_dict["loss"]
        return loss_dict

    def training_step(self, batch, batch_idx):
        # The batch dict now contains: prefix_inputs, teacher_answer_mask, student_inputs, student_labels, images
        # Pass all these to the forward method.
        try:
            loss_dict = self.forward(**batch)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not self._should_skip_oom(exc):
                raise
            self._handle_oom_skip("forward", batch_idx=batch_idx)
            return torch.zeros((), device=self.device, requires_grad=True)

        self.log(
            "train_ce_loss",
            loss_dict["ce_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_align_loss",
            loss_dict["align_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_alignment_gap",
            loss_dict["alignment_gap"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_total_loss",
            loss_dict["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self._update_epoch_train_metrics(loss_dict)

        if self.global_rank == 0:
            step_payload = {
                "epoch": int(self.current_epoch),
                "batch_idx": int(batch_idx),
                "global_step": int(self.global_step),
                "ce_loss": self._scalar_to_float(loss_dict["ce_loss"]),
                "align_loss": self._scalar_to_float(loss_dict["align_loss"]),
                "total_loss": self._scalar_to_float(loss_dict["total_loss"]),
                "alignment_gap": self._scalar_to_float(loss_dict["alignment_gap"]),
            }
            step_payload.update(self._current_vector_stats())
            self._append_jsonl(self._step_metrics_path(), step_payload)

        return loss_dict["loss"]

    def backward(self, loss, *args, **kwargs):
        try:
            return super().backward(loss, *args, **kwargs)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not self._should_skip_oom(exc):
                raise
            self._handle_oom_skip("backward")
            return None

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return

        self._write_epoch_train_metrics()

        if self.save_checkpoint_when(self.current_epoch):
            save_pretrained(
                os.path.join(self.save_dir, f"epoch-{self.current_epoch}"),
                self.lmm,
                self.shift_encoder,
            )

        eval_payload = self._run_validation_eval()
        if eval_payload is not None:
            self._save_eval_record(self.current_epoch, eval_payload)
            accuracy = float(eval_payload["eval_result"].get("accuracy", 0.0))
            alignment_gap = float(eval_payload["alignment_gap"])
            self.final_accuracy = accuracy
            self.final_epoch = int(self.current_epoch)
            self.epoch_eval_history.append(
                {
                    "epoch": int(self.current_epoch),
                    "accuracy": accuracy,
                    "alignment_gap": alignment_gap,
                    "record_path": self._record_eval_path(self.current_epoch),
                }
            )
            self._append_jsonl(
                self._eval_metrics_path(),
                {
                    "epoch": int(self.current_epoch),
                    "accuracy": accuracy,
                    "alignment_gap": alignment_gap,
                    "record_path": self._record_eval_path(self.current_epoch),
                    **eval_payload["trainable_vector_stats"],
                },
            )
            self.log(
                "val_accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = int(self.current_epoch)
                best_dir = os.path.join(self.save_dir, f"best-epoch-{self.current_epoch}")
                save_pretrained(best_dir, self.lmm, self.shift_encoder)
                self.best_checkpoint_path = best_dir
            print(
                "[EpochEval] "
                f"epoch={int(self.current_epoch)} "
                f"accuracy={accuracy:.4f} "
                f"alignment_gap={alignment_gap:.6f} "
                f"best_accuracy={self.best_accuracy:.4f} "
                f"record_path={self._record_eval_path(self.current_epoch)}"
            )
        else:
            print(
                "[EpochEval] "
                f"epoch={int(self.current_epoch)} skipped "
                f"(validation_eval.enabled={bool(getattr(getattr(self.cfg, 'validation_eval', None), 'enabled', False))})"
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank != 0:
            return

        current_step = int(self.global_step)
        if current_step == self._last_seen_global_step:
            return

        self._last_seen_global_step = current_step
        self._maybe_save_late_checkpoint()

    def on_train_end(self):
        if self.global_rank == 0:
            with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)
            self.final_checkpoint_path = os.path.join(self.save_dir, "final")
            save_pretrained(self.final_checkpoint_path, self.lmm, self.shift_encoder)
            self._write_summary()
            print(
                "[TrainEnd] "
                f"final_checkpoint_path={self.final_checkpoint_path} "
                f"best_accuracy={self.best_accuracy if self.best_accuracy != float('-inf') else None} "
                f"best_epoch={self.best_epoch} "
                f"summary_path={self._summary_path()}"
            )

    def configure_optimizers(self):
        def filter_decay_params(param_dict, **common_args):
            """filter parameters for optimizer, separate parameters by adding weight_decay or not"""
            non_decay_names = ["bias"]
            non_decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name in n
                    ],
                    "weight_decay": 0.0,
                    **common_args,
                }
            ]

            decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name not in n
                    ],
                    "weight_decay": self.cfg.weight_decay,
                    **common_args,
                }
            ]

            return [*non_decay, *decay]

        freeze_shift_scale = bool(getattr(self.cfg, "freeze_shift_scale", False))
        if freeze_shift_scale:
            for name, param in self.shift_encoder.named_parameters():
                if "shift_scale" in name:
                    param.requires_grad = False
            if self.global_rank == 0:
                print("[Optimizer] freeze_shift_scale=True, excluding *shift_scale* parameters from training.")

        param_dict = {
            n: p for n, p in self.shift_encoder.named_parameters() if p.requires_grad
        }
        if self.cfg.peft.get("scale_lr", None):
            # if scale_lr is provided, separate scale parameters and regular parameters
            # scale parameters will have a different learning rate, which typically is
            # used for LIVE.
            scale_params = {
                n: p for n, p in param_dict.items() if "log_Z1" in n or "scale" in n
            }
            regular_params = {
                n: p for n, p in param_dict.items() if n not in scale_params
            }

            optim_groups = [
                *filter_decay_params(regular_params, lr=self.cfg.lr),
                *filter_decay_params(scale_params, lr=self.cfg.peft.scale_lr),
            ]
        else:
            optim_groups = filter_decay_params(param_dict, lr=self.cfg.lr)

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if "deepspeed" in self.cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
