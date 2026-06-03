import gc
import math
import os
import re
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import paths  # noqa: F401
from dataset_utils import dataset_mapping
from extract_cot_vector_internlm import (
    TokenizerProcessorAdapter,
    build_language_token_mask,
    build_student_answer_mask,
    build_teacher_answer_mask,
    decode_masked_text,
    extract_sample_attn_tokens,
    iter_safe_batches,
    log_alignment_mismatch,
    remove_hook_dict,
)
from qwen_model_wrapper import QwenModelWrapper


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def internvl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    module_name=None,
    recorder=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if module_name is None:
        raise ValueError("module_name is required for InternVL attention extraction.")
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    if recorder is not None:
        if recorder.current_recording_context == "teacher":
            recorder.teacher_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()
        elif recorder.current_recording_context == "student":
            recorder.student_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


class InternVLExtractionRecorder:
    def __init__(self, lmm):
        self.lmm = lmm
        self.model = lmm.model
        self.lm_model = lmm.model.language_model
        self.model_name = lmm.model_name.lower()
        self.num_layers = self.lm_model.config.num_hidden_layers
        self.hidden_size = self.lm_model.config.hidden_size
        self.num_heads = self.lm_model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.current_recording_context = None
        self.teacher_raw_attn_outputs = [None for _ in range(self.num_layers)]
        self.student_raw_attn_outputs = [None for _ in range(self.num_layers)]
        self.ffn_hidden_states = [None for _ in range(self.num_layers)]

    def reset_hidden_states(self):
        self.teacher_raw_attn_outputs = [None for _ in range(self.num_layers)]
        self.student_raw_attn_outputs = [None for _ in range(self.num_layers)]
        self.ffn_hidden_states = [None for _ in range(self.num_layers)]
        self.current_recording_context = None

    def register_record_hooks(self):
        def ffn_record_hook(module, inputs, outputs, module_name, **kwargs):
            layer_idx = int(module_name.split(".")[3])
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.ffn_hidden_states[layer_idx] = hidden_states

        return {
            "ffn_record_hook": self.lmm.register_forward_hook(
                r"language_model\.model\.layers\.\d+\.feed_forward$",
                ffn_record_hook,
            )
        }

    def register_attention_forward(self):
        if getattr(self, "_attention_forward_replaced", False):
            return
        self.lmm.replace_module_method(
            r"language_model\.model\.layers\.\d+\.attention$",
            "forward",
            partial(internvl_attn_forward, recorder=self),
            strict=False,
        )
        self._attention_forward_replaced = True


@hydra.main(config_path="config", config_name="extract_cot_vector_internvl.yaml", version_base=None)
def main(cfg: DictConfig):
    device_idx = int(getattr(cfg, "devices", 0))
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {cfg.model_name} on {device}")

    lmm = QwenModelWrapper(
        model_root=cfg.model_path,
        processor_class=AutoTokenizer,
        model_class=AutoModel,
        support_models=[cfg.model_name],
        local_files_only=True,
        torch_dtype=eval(cfg.dtype),
        processor_args={"padding_side": "left", "use_fast": False},
        model_args={"output_hidden_states": True},
        model_name=cfg.model_name,
    ).to(device)

    if lmm.processor.pad_token is None:
        lmm.processor.pad_token = lmm.processor.eos_token
    lmm.processor = TokenizerProcessorAdapter(lmm.processor)

    if getattr(cfg, "offload_vision_to_cpu", True):
        if hasattr(lmm.model, "vision_model"):
            lmm.model.vision_model.to("cpu")
        if hasattr(lmm.model, "mlp1"):
            lmm.model.mlp1.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lmm.eval()
    for param in lmm.model.parameters():
        param.requires_grad = False

    cfg.data.training_mode = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    if not hasattr(cfg.data, "train_use_images"):
        with open_dict(cfg.data):
            cfg.data.train_use_images = False
    print(f"InternVL extract train_use_images={cfg.data.train_use_images}")
    dataset = dataset_mapping[cfg.data.name](cfg.data, model_processor=lmm.processor, model_name=cfg.model_name)

    shard_count = int(getattr(cfg, "shard_count", 1) or 1)
    shard_index = int(getattr(cfg, "shard_index", 0) or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}.")
        shard_indices = list(range(shard_index, len(dataset._support_set), shard_count))
        dataset._support_set = dataset._support_set.select(shard_indices)
        print(
            f"Extracting shard {shard_index + 1}/{shard_count} with "
            f"{len(dataset._support_set)} support samples."
        )

    dataloader = dataset.train_dataloader(lmm, batch_size=cfg.batch_size)

    recorder = InternVLExtractionRecorder(lmm=lmm)
    recorder.register_attention_forward()

    num_layers = lmm.model.language_model.config.num_hidden_layers
    hidden_size = lmm.model.language_model.config.hidden_size
    num_heads = lmm.model.language_model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    accumulated_ffn_gaps = [torch.zeros(hidden_size, device=device) for _ in range(num_layers)]
    accumulated_ffn_hs = [torch.zeros(hidden_size, device=device) for _ in range(num_layers)]
    ffn_sample_count_per_layer = [0 for _ in range(num_layers)]

    accumulated_attn_gaps = [
        torch.zeros(num_heads, head_dim, device=device) for _ in range(num_layers)
    ]
    attn_sample_count_per_layer = [0 for _ in range(num_layers)]

    pad_id = lmm.processor.tokenizer.pad_token_id
    eos_id = lmm.processor.tokenizer.eos_token_id

    print("Starting InternVL CoT vector extraction with answer-region averaging...")

    skipped_sample_count = 0
    dataloader_batch_size = dataloader.batch_size or cfg.batch_size or 1
    total_batches = (len(dataloader.dataset) + dataloader_batch_size - 1) // dataloader_batch_size

    for step, (batch, skipped_samples) in enumerate(
        tqdm(
            iter_safe_batches(dataloader, seed=getattr(cfg.data, "seed", None)),
            total=total_batches,
            desc="Extracting CoT vectors",
        )
    ):
        if skipped_samples:
            skipped_sample_count += len(skipped_samples)
        if batch is None:
            print(f"Warning: skipped empty batch at step {step} after removing invalid samples.")
            continue

        recorder.reset_hidden_states()
        recorder.current_recording_context = "teacher"
        hooks_teacher = recorder.register_record_hooks()
        with torch.no_grad():
            teacher_inputs = {
                k: v.to(device)
                for k, v in batch["prefix_inputs"].items()
                if isinstance(v, torch.Tensor)
            }
            _ = lmm.model.language_model(**teacher_inputs, output_hidden_states=True, use_cache=False)
            teacher_raw_attn_hs = [
                t.clone().to(device) if t is not None else None for t in recorder.teacher_raw_attn_outputs
            ]
            teacher_ffn_hs = [
                t.clone().to(device) if t is not None else None for t in recorder.ffn_hidden_states
            ]
        remove_hook_dict(hooks_teacher)

        recorder.reset_hidden_states()
        recorder.current_recording_context = "student"
        hooks_student = recorder.register_record_hooks()
        with torch.no_grad():
            student_inputs = {
                k: v.to(device)
                for k, v in batch["student_inputs"].items()
                if isinstance(v, torch.Tensor)
            }
            _ = lmm.model.language_model(**student_inputs, output_hidden_states=True, use_cache=False)
            student_raw_attn_hs = [
                t.clone().to(device) if t is not None else None for t in recorder.student_raw_attn_outputs
            ]
            student_ffn_hs = [
                t.clone().to(device) if t is not None else None for t in recorder.ffn_hidden_states
            ]
        remove_hook_dict(hooks_student)

        student_answer_mask = build_student_answer_mask(batch=batch, device=device)
        teacher_answer_mask = build_teacher_answer_mask(
            batch=batch,
            device=device,
            tokenizer=lmm.processor.tokenizer,
        )

        teacher_ids = teacher_inputs["input_ids"]
        student_ids = student_inputs["input_ids"]
        teacher_final_mask = teacher_answer_mask & build_language_token_mask(
            teacher_ids,
            teacher_inputs.get("attention_mask"),
            pad_id=pad_id,
            eos_id=eos_id,
        )
        student_final_mask = student_answer_mask & build_language_token_mask(
            student_ids,
            student_inputs.get("attention_mask"),
            pad_id=pad_id,
            eos_id=eos_id,
        )

        if step == 0:
            tok = lmm.processor.tokenizer
            teacher_ids_cpu = teacher_ids[0].detach().cpu()
            student_ids_cpu = student_ids[0].detach().cpu()
            teacher_final_mask_cpu = teacher_final_mask[0].detach().cpu()
            student_final_mask_cpu = student_final_mask[0].detach().cpu()
            print("\n===== DEBUG FIRST SAMPLE =====")
            print("\nTeacher FULL TEXT:")
            print(tok.decode(teacher_ids_cpu, skip_special_tokens=False))
            print("\nStudent FULL TEXT:")
            print(tok.decode(student_ids_cpu, skip_special_tokens=False))
            print("\nTeacher final mask text:")
            print(decode_masked_text(tok, teacher_ids_cpu, teacher_final_mask_cpu))
            print("\nStudent final mask text:")
            print(decode_masked_text(tok, student_ids_cpu, student_final_mask_cpu))
            print("\nTeacher final token count:", teacher_final_mask_cpu.sum().item())
            print("Student final token count:", student_final_mask_cpu.sum().item())
            print("\n=============================\n")

        batch_size = teacher_ids.shape[0]
        tok = lmm.processor.tokenizer
        for layer_idx in range(num_layers):
            t_f_hs = teacher_ffn_hs[layer_idx]
            s_f_hs = student_ffn_hs[layer_idx]
            t_a_hs = teacher_raw_attn_hs[layer_idx]
            s_a_hs = student_raw_attn_hs[layer_idx]

            for sample_idx in range(batch_size):
                sample_t_mask = teacher_final_mask[sample_idx]
                sample_s_mask = student_final_mask[sample_idx]

                teacher_token_count = int(sample_t_mask.sum().item())
                student_token_count = int(sample_s_mask.sum().item())

                if t_f_hs is not None and s_f_hs is not None:
                    sample_t_ffn = t_f_hs[sample_idx][sample_t_mask]
                    sample_s_ffn = s_f_hs[sample_idx][sample_s_mask]

                    if sample_t_ffn.numel() == 0 or sample_s_ffn.numel() == 0:
                        if teacher_token_count != student_token_count:
                            teacher_text = decode_masked_text(
                                tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu()
                            )
                            student_text = decode_masked_text(
                                tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu()
                            )
                            log_alignment_mismatch(
                                "FFN",
                                step,
                                sample_idx,
                                layer_idx,
                                teacher_token_count,
                                student_token_count,
                                teacher_text,
                                student_text,
                            )
                    elif sample_t_ffn.shape == sample_s_ffn.shape:
                        ffn_gap = sample_t_ffn - sample_s_ffn
                        accumulated_ffn_gaps[layer_idx] += ffn_gap.mean(dim=0)
                        accumulated_ffn_hs[layer_idx] += sample_t_ffn.mean(dim=0)
                        ffn_sample_count_per_layer[layer_idx] += 1
                    else:
                        teacher_text = decode_masked_text(
                            tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu()
                        )
                        student_text = decode_masked_text(
                            tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu()
                        )
                        log_alignment_mismatch(
                            "FFN",
                            step,
                            sample_idx,
                            layer_idx,
                            teacher_token_count,
                            student_token_count,
                            teacher_text,
                            student_text,
                        )

                if t_a_hs is not None and s_a_hs is not None:
                    try:
                        sample_t_attn = extract_sample_attn_tokens(
                            t_a_hs, sample_idx, sample_t_mask, num_heads, head_dim
                        )
                        sample_s_attn = extract_sample_attn_tokens(
                            s_a_hs, sample_idx, sample_s_mask, num_heads, head_dim
                        )

                        if sample_t_attn.numel() == 0 or sample_s_attn.numel() == 0:
                            if teacher_token_count != student_token_count:
                                teacher_text = decode_masked_text(
                                    tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu()
                                )
                                student_text = decode_masked_text(
                                    tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu()
                                )
                                log_alignment_mismatch(
                                    "Attention",
                                    step,
                                    sample_idx,
                                    layer_idx,
                                    teacher_token_count,
                                    student_token_count,
                                    teacher_text,
                                    student_text,
                                )
                        elif sample_t_attn.shape == sample_s_attn.shape:
                            attn_gap = sample_t_attn - sample_s_attn
                            accumulated_attn_gaps[layer_idx] += attn_gap.mean(dim=0)
                            attn_sample_count_per_layer[layer_idx] += 1
                        else:
                            teacher_text = decode_masked_text(
                                tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu()
                            )
                            student_text = decode_masked_text(
                                tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu()
                            )
                            log_alignment_mismatch(
                                "Attention",
                                step,
                                sample_idx,
                                layer_idx,
                                teacher_token_count,
                                student_token_count,
                                teacher_text,
                                student_text,
                            )
                    except Exception as exc:
                        teacher_text = decode_masked_text(
                            tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu()
                        )
                        student_text = decode_masked_text(
                            tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu()
                        )
                        print(
                            f"Warning: attention extraction failed at step {step}, "
                            f"sample {sample_idx}, layer {layer_idx}: {exc}"
                        )
                        print(f"Teacher masked text: {teacher_text}")
                        print(f"Student masked text: {student_text}")

        del teacher_raw_attn_hs, teacher_ffn_hs, student_raw_attn_hs, student_ffn_hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished extraction with {skipped_sample_count} skipped samples.")

    final_ffn_cot_vector = torch.stack(
        [vec / count if count > 0 else torch.zeros_like(vec) for vec, count in zip(accumulated_ffn_gaps, ffn_sample_count_per_layer)]
    )
    final_ffn_hs_vector = torch.stack(
        [vec / count if count > 0 else torch.zeros_like(vec) for vec, count in zip(accumulated_ffn_hs, ffn_sample_count_per_layer)]
    )
    final_attn_cot_vector = torch.stack(
        [vec / count if count > 0 else torch.zeros_like(vec) for vec, count in zip(accumulated_attn_gaps, attn_sample_count_per_layer)]
    )

    output_dir = os.path.dirname(cfg.output_path)
    os.makedirs(output_dir, exist_ok=True)

    encoder_cfg = OmegaConf.create(
        {
            "_target_": "src.extract_cot_vector_internvl.InternVLExtractionRecorder",
            "attn_strategy": "raw_attn_gap",
            "ffn_strategy": "ffn_output_gap",
        }
    )
    cot_vectors_to_save = {
        "ffn_cot_vector": final_ffn_cot_vector.cpu(),
        "ffn_hs_vector": final_ffn_hs_vector.cpu(),
        "attn_cot_vector": final_attn_cot_vector.cpu(),
        "ffn_cot_vector_sums": torch.stack(accumulated_ffn_gaps).cpu(),
        "ffn_hs_vector_sums": torch.stack(accumulated_ffn_hs).cpu(),
        "attn_cot_vector_sums": torch.stack(accumulated_attn_gaps).cpu(),
        "ffn_sample_count_per_layer": torch.tensor(ffn_sample_count_per_layer, dtype=torch.long),
        "attn_sample_count_per_layer": torch.tensor(attn_sample_count_per_layer, dtype=torch.long),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "processed_support_samples": len(dataloader.dataset),
        "skipped_sample_count": skipped_sample_count,
        "encoder_type": encoder_cfg,
        "multi_head_attn_strategy": True,
    }

    torch.save(cot_vectors_to_save, cfg.output_path)
    print(f"Average CoT vectors saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
