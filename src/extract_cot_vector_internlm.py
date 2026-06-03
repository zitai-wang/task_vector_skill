import gc
import math
import os
import random
import re
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import paths  # noqa: F401
from dataset_utils import dataset_mapping
from qwen_model_wrapper import QwenModelWrapper
from utils import build_model


ANSWER_SPAN_ERROR_RE = re.compile(
    r"Failed to locate teacher answer span for batch sample (?P<sample_idx>\d+)"
)


def remove_hook_dict(hook_dict):
    if hook_dict is None:
        return
    for _, hooks in hook_dict.items():
        if isinstance(hooks, list):
            for hook in hooks:
                if hook is not None:
                    hook.remove()
        elif hooks is not None:
            hooks.remove()


def build_student_answer_mask(batch, device):
    if "student_answer_mask" in batch:
        return batch["student_answer_mask"].to(device).bool()

    if "student_labels" in batch:
        return batch["student_labels"].to(device).ne(-100)

    raise KeyError(
        "Missing `student_answer_mask` in batch. "
        "Extraction requires dataset to return a strict answer-region mask."
    )


def build_teacher_answer_mask(batch, device, tokenizer=None):
    if "teacher_answer_mask" not in batch:
        raise KeyError(
            "Missing `teacher_answer_mask` in batch. "
            "Extraction requires dataset to return a strict answer-region mask."
        )

    teacher_mask = batch["teacher_answer_mask"].to(device).bool()
    if teacher_mask.dim() != 2:
        return teacher_mask

    if "prefix_inputs" not in batch or "student_labels" not in batch:
        return teacher_mask

    teacher_input_ids = batch["prefix_inputs"]["input_ids"].to(device)
    student_labels = batch["student_labels"].to(device)
    pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    for sample_idx in range(teacher_mask.shape[0]):
        if teacher_mask[sample_idx].any():
            continue

        answer_ids = student_labels[sample_idx][student_labels[sample_idx].ne(-100)]
        if answer_ids.numel() == 0:
            continue

        if pad_id is not None:
            answer_ids = answer_ids[answer_ids.ne(pad_id)]
        if eos_id is not None:
            answer_ids = answer_ids[answer_ids.ne(eos_id)]
        if answer_ids.numel() == 0:
            continue

        search_candidates = [answer_ids]
        if answer_ids.numel() > 1:
            search_candidates.append(answer_ids[:-1])

        if tokenizer is not None:
            answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
            if answer_text:
                for text_variant in (
                    answer_text,
                    " " + answer_text,
                    "\n" + answer_text,
                    "#### " + answer_text,
                ):
                    variant_ids = tokenizer(text_variant, add_special_tokens=False).input_ids
                    if variant_ids:
                        search_candidates.append(
                            torch.tensor(variant_ids, device=device, dtype=teacher_input_ids.dtype)
                        )

        full_ids = teacher_input_ids[sample_idx]
        matched = False
        for candidate in search_candidates:
            cand_len = candidate.numel()
            if cand_len == 0 or cand_len > full_ids.numel():
                continue
            for start in range(full_ids.numel() - cand_len, -1, -1):
                if torch.equal(full_ids[start : start + cand_len], candidate):
                    teacher_mask[sample_idx, start : start + cand_len] = True
                    matched = True
                    break
            if matched:
                break

    return teacher_mask


def decode_masked_text(tokenizer, input_ids, mask):
    if mask.numel() == 0 or not mask.any():
        return ""
    return tokenizer.decode(input_ids[mask].tolist(), skip_special_tokens=False)


def log_alignment_mismatch(kind, step, sample_idx, layer_idx, teacher_tokens, student_tokens, teacher_text, student_text):
    print(
        f"Warning: {kind} token mismatch at step {step}, sample {sample_idx}, layer {layer_idx}: "
        f"teacher_tokens={teacher_tokens}, student_tokens={student_tokens}"
    )
    print(f"Teacher masked text: {teacher_text}")
    print(f"Student masked text: {student_text}")


def extract_sample_attn_tokens(attn_hs, sample_idx, token_mask, num_heads, head_dim):
    sample_attn = attn_hs[sample_idx]

    if sample_attn.dim() != 3:
        raise ValueError(f"Unexpected sample attention rank: {tuple(sample_attn.shape)}")

    if sample_attn.shape[0] == num_heads and sample_attn.shape[2] == head_dim:
        return sample_attn[:, token_mask, :].permute(1, 0, 2).contiguous()

    if sample_attn.shape[1] == num_heads and sample_attn.shape[2] == head_dim:
        return sample_attn[token_mask].contiguous()

    raise ValueError(
        f"Cannot infer sample attention layout: {tuple(sample_attn.shape)}, "
        f"expected num_heads={num_heads}, head_dim={head_dim}"
    )


def build_language_token_mask(input_ids, attention_mask, pad_id, eos_id):
    if attention_mask is None:
        lang_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        lang_mask = attention_mask.bool()

    if pad_id is not None:
        lang_mask &= input_ids.ne(pad_id)
    if eos_id is not None:
        lang_mask &= input_ids.ne(eos_id)
    return lang_mask


def collate_batch_with_bad_sample_retry(collate_fn, batch_samples, batch_indices):
    pending_samples = list(batch_samples)
    pending_indices = list(batch_indices)
    skipped_samples = []

    while pending_samples:
        try:
            return collate_fn(pending_samples), skipped_samples
        except ValueError as exc:
            message = str(exc)
            match = ANSWER_SPAN_ERROR_RE.search(message)
            if match is None:
                raise

            failed_pos = int(match.group("sample_idx"))
            if failed_pos < 0 or failed_pos >= len(pending_samples):
                raise

            failed_sample = pending_samples.pop(failed_pos)
            failed_index = pending_indices.pop(failed_pos)
            failed_answer = None
            if isinstance(failed_sample, dict):
                failed_answer = failed_sample.get("gt_numerical", failed_sample.get("answer"))

            skipped_samples.append(
                {
                    "dataset_index": failed_index,
                    "answer": failed_answer,
                    "error": message,
                }
            )
            print(
                "Warning: skipping sample with unmatched teacher answer span "
                f"(dataset_index={failed_index}, answer={failed_answer!r})."
            )

    return None, skipped_samples


def iter_safe_batches(dataloader, seed=None):
    data_source = dataloader.dataset
    collate_fn = dataloader.collate_fn
    batch_size = dataloader.batch_size or 1

    indices = list(range(len(data_source)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_samples = [data_source[idx] for idx in batch_indices]
        batch, skipped_samples = collate_batch_with_bad_sample_retry(
            collate_fn=collate_fn,
            batch_samples=batch_samples,
            batch_indices=batch_indices,
        )
        yield batch, skipped_samples


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


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TokenizerProcessorAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class InternLMExtractionRecorder:
    def __init__(self, lmm):
        self.lmm = lmm
        self.model = lmm.model
        self.model_name = lmm.model_name.lower()
        self.num_layers = lmm.model.config.num_hidden_layers
        self.hidden_size = lmm.model.config.hidden_size
        self.num_heads = lmm.model.config.num_attention_heads
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
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.ffn_hidden_states[layer_idx] = hidden_states

        return {
            "ffn_record_hook": self.lmm.register_forward_hook(
                r"model\.layers\.\d+\.feed_forward$",
                ffn_record_hook,
            )
        }

    def register_attention_forward(self):
        if getattr(self, "_attention_forward_replaced", False):
            return
        self.lmm.replace_module_method(
            r"model\.layers\.\d+\.attention$",
            "forward",
            partial(internlm_attn_forward, recorder=self),
            strict=False,
        )
        self._attention_forward_replaced = True


def internlm_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    recorder=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        qkv_slices = self.wqkv.weight.split(key_value_slicing, dim=0)
        qkv_states = torch.cat(
            [nn.functional.linear(hidden_states, qkv_slice) for qkv_slice in qkv_slices],
            dim=-1,
        )
    else:
        qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d").transpose(1, 2)
    key_states = qkv_states[..., -2, :].transpose(1, 2)
    value_states = qkv_states[..., -1, :].transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if module_name is None:
        raise ValueError("module_name is required for InternLM attention extraction.")
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    if recorder is not None:
        if recorder.current_recording_context == "teacher":
            recorder.teacher_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()
        elif recorder.current_recording_context == "student":
            recorder.student_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.wo.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            [nn.functional.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
        )
    else:
        attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


@hydra.main(config_path="config", config_name="extract_cot_vector_internlm.yaml", version_base=None)
def main(cfg: DictConfig):
    device_idx = int(getattr(cfg, "devices", 0))
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {cfg.model_name} on {device}")

    model_hf_id, model_path = build_model(cfg)
    lmm = QwenModelWrapper(
        model_root=model_path,
        processor_class=AutoTokenizer,
        model_class=AutoModelForCausalLM,
        support_models=[model_hf_id],
        local_files_only=True,
        torch_dtype=eval(cfg.dtype),
        processor_args={"padding_side": "left", "use_fast": False},
        model_args={"output_hidden_states": True},
        model_name=cfg.model_name,
    ).to(device)

    if lmm.processor.pad_token is None:
        lmm.processor.pad_token = lmm.processor.eos_token
    lmm.processor = TokenizerProcessorAdapter(lmm.processor)

    lmm.eval()
    for param in lmm.model.parameters():
        param.requires_grad = False

    cfg.data.training_mode = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
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

    recorder = InternLMExtractionRecorder(lmm=lmm)
    recorder.register_attention_forward()

    num_layers = lmm.model.config.num_hidden_layers
    hidden_size = lmm.model.config.hidden_size
    num_heads = lmm.model.config.num_attention_heads
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

    print("Starting InternLM CoT vector extraction with answer-region averaging...")

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
            _ = lmm.model(**teacher_inputs, output_hidden_states=True, use_cache=False)
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
            _ = lmm.model(**student_inputs, output_hidden_states=True, use_cache=False)
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
            "_target_": "src.extract_cot_vector_internlm.InternLMExtractionRecorder",
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
