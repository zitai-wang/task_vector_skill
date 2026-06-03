import ast
import enum
import glob
import io
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils.interface import DatasetBase


class DatasetState(enum.Enum):
    TRAIN_TEACHER = "TRAIN_TEACHER"
    TRAIN_STUDENT_ONESHOT = "TRAIN_STUDENT_ONESHOT"
    TRAIN_STUDENT_DIRECT_Q = "TRAIN_STUDENT_DIRECT_Q"
    TRAIN_TEACHER_SELF_COT = "TRAIN_TEACHER_SELF_COT"
    TRAIN_STUDENT_ONESHOT_SELF_COT = "TRAIN_STUDENT_ONESHOT_SELF_COT"
    TRAIN_STUDENT_DIRECT_Q_SELF_COT = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    EVAL_BASELINE = "EVAL_BASELINE"
    EVAL_WITH_COT_VECTOR_ONESHOT = "EVAL_WITH_COT_VECTOR_ONESHOT"
    EVAL_WITH_COT_VECTOR_DIRECT_Q = "EVAL_WITH_COT_VECTOR_DIRECT_Q"


class MMMUDataset(DatasetBase):
    support_datasets = ["mmmu"]

    def __init__(
        self,
        data_cfg: DictConfig,
        model_processor: Any = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        self.data_cfg = data_cfg
        self.dataset_state = DatasetState.EVAL_BASELINE
        self.skip_overlength_samples = bool(
            getattr(data_cfg, "skip_overlength_samples", False)
        )

        self.source = getattr(data_cfg, "source", "MMMU/MMMU")
        self.cache_dir = getattr(data_cfg, "cache_dir", None)
        self.is_local_source = os.path.isdir(self.source)
        self.query_split = getattr(data_cfg, "query_split", getattr(data_cfg, "split", "validation"))
        self.support_split = getattr(data_cfg, "support_split", "dev")
        self.subsets = self._resolve_subsets()

        self._query_set = self._load_split(self.query_split)
        try:
            self._support_set = self._load_split(self.support_split)
        except Exception as exc:
            print(
                f"Warning: failed to load MMMU support split '{self.support_split}': {exc}. "
                f"Falling back to query split '{self.query_split}'."
            )
            self._support_set = self._query_set

        self._self_cot_data = None
        if hasattr(data_cfg, "use_self_cot") and data_cfg.use_self_cot:
            self._load_self_cot_data(data_cfg)

        support_limit = getattr(
            data_cfg,
            "num_support_samples",
            getattr(data_cfg, "num_query_samples", 0),
        )
        if support_limit and support_limit > 0:
            num = min(int(support_limit), len(self._support_set))
            rng = random.Random(getattr(data_cfg, "seed", None))
            indices = rng.sample(range(len(self._support_set)), num)
            self._support_set = self._support_set.select(indices)
            print(
                f"MMMU initialized: Support set limited to {len(self._support_set)} samples "
                f"with seed={getattr(data_cfg, 'seed', None)}."
            )

        self.direct_answer_system_prompt = (
            "You are a careful multimodal expert assistant. "
            "For multiple-choice questions, answer with the option letter only. "
            "For open questions, answer with a short final response."
        )
        self.cot_system_prompt = (
            "You are a careful multimodal expert assistant. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    def _resample_single_train_sample(self, max_seq_len: int, reason: str):
        if len(self._support_set) <= 1:
            raise ValueError(
                f"Cannot resample MMMU train sample after failure: {reason}. "
                "Support set has <= 1 sample."
            )

        max_resample_attempts = 20
        for _ in range(max_resample_attempts):
            replacement = self._support_set[random.randrange(len(self._support_set))]
            try:
                return self.collate_fn_for_train([replacement], max_seq_len)
            except ValueError as exc:
                msg = str(exc)
                if (
                    "exceeds max_seq_len" in msg
                    or "Failed to locate teacher answer span" in msg
                ):
                    continue
                raise

        raise ValueError(
            f"MMMU failed to resample a valid train sample within {max_resample_attempts} attempts. "
            f"Last failure reason: {reason}"
        )

    @staticmethod
    def metric_key() -> str:
        return "accuracy"

    @property
    def instruction(self) -> str:
        return ""

    @property
    def num_role_in_round(self) -> int:
        return 2

    def _resolve_subsets(self) -> List[str]:
        explicit_subsets = getattr(self.data_cfg, "subsets", None)
        if explicit_subsets:
            return self._coerce_string_list(explicit_subsets)

        hf_name = getattr(self.data_cfg, "hf_name", None)
        if hf_name:
            return [str(hf_name)]

        if self.is_local_source:
            return [
                name
                for name in sorted(os.listdir(self.source))
                if os.path.isdir(os.path.join(self.source, name)) and not name.startswith(".")
            ]

        try:
            config_names = get_dataset_config_names(self.source)
        except Exception as exc:
            raise ValueError(
                "Unable to infer MMMU subject configs automatically. "
                "Please set `data.subsets=[...]` or `data.hf_name=...` explicitly."
            ) from exc

        return [name for name in config_names if name]

    @staticmethod
    def _coerce_string_list(value: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            return [part.strip() for part in stripped.split(",") if part.strip()]

        return [str(item).strip() for item in value if str(item).strip()]

    def _load_split(self, split: str) -> Dataset:
        split_datasets = []
        for subset in self.subsets:
            if self.is_local_source:
                pattern = os.path.join(self.source, subset, f"{split}-*.parquet")
                files = sorted(glob.glob(pattern))
                if not files:
                    continue
                ds = Dataset.from_parquet(files, cache_dir=self.cache_dir)
            else:
                ds = load_dataset(
                    self.source,
                    name=subset,
                    split=split,
                    cache_dir=self.cache_dir,
                )
            if "subject" not in ds.column_names:
                ds = ds.add_column("subject", [subset] * len(ds))
            split_datasets.append(ds)

        if not split_datasets:
            raise ValueError(f"No MMMU subsets were loaded for split='{split}'.")

        if len(split_datasets) == 1:
            return split_datasets[0]
        return concatenate_datasets(split_datasets)

    def _load_self_cot_data(self, data_cfg: DictConfig):
        path = getattr(data_cfg, "self_cot_path", None)
        if not path or not os.path.exists(path):
            print(f"Warning: Self-CoT path not found at {path}")
            return

        self._self_cot_data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pid = record.get("pid", record.get("id"))
                if pid is not None:
                    self._self_cot_data[str(pid)] = record

        filtered_indices = []
        for idx, sample in enumerate(self._support_set):
            pid = str(sample.get("id"))
            record = self._self_cot_data.get(pid)
            if not record:
                continue
            if "is_correct" in record and not record.get("is_correct", False):
                continue
            self_cot = record.get("self_cot", "")
            if not isinstance(self_cot, str) or not self_cot.strip():
                continue
            filtered_indices.append(idx)

        if filtered_indices:
            self._support_set = self._support_set.select(filtered_indices)
            print(f"MMMU: Loaded {len(self._support_set)} validated samples from Self-CoT data.")
        else:
            print("Warning: No valid MMMU Self-CoT samples found after filtering.")

    @staticmethod
    def _normalize_question_text(question: str) -> str:
        text = str(question or "")
        text = re.sub(r"<image\s*\d+>", "<image>", text, flags=re.IGNORECASE)
        text = re.sub(r"<image\d+>", "<image>", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _parse_options(options: Any) -> List[str]:
        if options is None:
            return []

        if isinstance(options, dict):
            return [str(options[key]).strip() for key in sorted(options.keys())]

        if isinstance(options, (list, tuple)):
            return [str(opt).strip() for opt in options]

        text = str(options).strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return [str(parsed[key]).strip() for key in sorted(parsed.keys())]
            if isinstance(parsed, (list, tuple)):
                return [str(opt).strip() for opt in parsed]
        except Exception:
            pass

        if "|||" in text:
            return [part.strip() for part in text.split("|||") if part.strip()]

        return [text]

    @classmethod
    def _question_type(cls, item: Dict) -> str:
        qtype = str(item.get("question_type", "")).strip().lower()
        if "multiple" in qtype:
            return "multiple-choice"

        options = cls._parse_options(item.get("options"))
        if options:
            return "multiple-choice"
        return "open"

    @staticmethod
    def _decode_image_dict(image_dict: Dict[str, Any]) -> Optional[Image.Image]:
        img_path = image_dict.get("path")
        if img_path and os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")

        img_bytes = image_dict.get("bytes")
        if img_bytes:
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return None

    def _collect_images(self, item: Dict) -> List[Image.Image]:
        image_items: List[Tuple[int, Any]] = []
        for key, value in item.items():
            if key == "image":
                image_items.append((0, value))
                continue

            match = re.fullmatch(r"image_(\d+)", key)
            if match:
                image_items.append((int(match.group(1)), value))

        images: List[Image.Image] = []
        for _, image_value in sorted(image_items, key=lambda x: x[0]):
            if image_value is None:
                continue
            if isinstance(image_value, Image.Image):
                images.append(image_value.convert("RGB"))
                continue
            if isinstance(image_value, dict):
                decoded = self._decode_image_dict(image_value)
                if decoded is not None:
                    images.append(decoded)
                continue
            if isinstance(image_value, str) and os.path.exists(image_value):
                images.append(Image.open(image_value).convert("RGB"))

        return images

    @staticmethod
    def _list_to_dict(options: List[str]) -> Dict[str, str]:
        return {chr(ord("A") + idx): str(option) for idx, option in enumerate(options)}

    @staticmethod
    def _strip_prediction_wrappers(text: str) -> str:
        candidate = str(text or "").strip()
        if not candidate:
            return ""

        answer_tag = re.findall(r"<answer>(.*?)</answer>", candidate, re.DOTALL | re.IGNORECASE)
        if answer_tag:
            return answer_tag[-1].strip()

        boxed = re.findall(r"\\boxed\{([^}]*)\}", candidate)
        if boxed:
            return boxed[-1].strip()

        explicit_answer = re.findall(
            r"(?:Final Answer|Answer|answer)\s*[:：]\s*([^\n]+)",
            candidate,
        )
        if explicit_answer:
            return explicit_answer[-1].strip()

        return candidate

    @classmethod
    def _canonical_answer_text(cls, answer: Any) -> str:
        parsed = cls._parse_answer_field(answer)
        if isinstance(parsed, list):
            for value in parsed:
                text = str(value).strip()
                if text:
                    return text
            return ""
        return str(parsed).strip()

    @staticmethod
    def _parse_answer_field(answer: Any) -> Union[str, List[str]]:
        if isinstance(answer, (list, tuple)):
            return [str(item).strip() for item in answer]

        text = str(answer).strip()
        if not text:
            return ""

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed]
        except Exception:
            pass
        return text

    @staticmethod
    def _normalize_open_string(text: Any) -> str:
        value = str(text).strip().lower()
        value = value.strip(".")
        value = value.strip()
        value = value.strip("\"'")
        return value

    @classmethod
    def _extract_numeric_candidates(cls, text: str) -> List[str]:
        matches = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(text), flags=re.IGNORECASE)
        normalized = []
        for match in matches:
            try:
                number = float(match.replace(",", ""))
                if number.is_integer():
                    normalized.append(str(int(number)))
                normalized.append(str(number))
            except Exception:
                continue
        return normalized

    @classmethod
    def _normalize_open_candidates(cls, value: Any) -> List[str]:
        candidates: List[str] = []
        if isinstance(value, (list, tuple)):
            for item in value:
                candidates.extend(cls._normalize_open_candidates(item))
            return cls._dedupe_keep_order(candidates)

        text = cls._normalize_open_string(value)
        if text:
            candidates.append(text)
        candidates.extend(cls._extract_numeric_candidates(text))
        return cls._dedupe_keep_order(candidates)

    @staticmethod
    def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
        seen = set()
        unique_values = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    @classmethod
    def _parse_open_response(cls, response: str) -> List[str]:
        raw = str(response or "").strip()
        if not raw:
            return []

        key_responses = [cls._strip_prediction_wrappers(raw)]
        normalized_raw = raw.replace("\n", " \n ").strip()

        indicators = [
            "final answer",
            "answer",
            "thus",
            "therefore",
            "so",
            "result",
        ]
        lower_raw = normalized_raw.lower()
        for indicator in indicators:
            index = lower_raw.rfind(indicator)
            if index != -1:
                tail = normalized_raw[index + len(indicator):].strip(" :.-\n")
                if tail:
                    key_responses.append(tail)

        key_responses.extend([line.strip() for line in raw.splitlines() if line.strip()])
        key_responses.append(raw)

        predictions = []
        for candidate in key_responses:
            predictions.extend(cls._normalize_open_candidates(candidate))
        return cls._dedupe_keep_order(predictions)

    @classmethod
    def _normalize_gold_choices(cls, answer: Any, choices: Dict[str, str]) -> List[str]:
        parsed = cls._parse_answer_field(answer)
        answers = parsed if isinstance(parsed, list) else [parsed]
        normalized: List[str] = []
        for value in answers:
            text = str(value).strip()
            if not text:
                continue

            upper = text.upper()
            if upper in choices:
                normalized.append(upper)
                continue

            for key, choice_text in choices.items():
                if text.lower() == str(choice_text).strip().lower():
                    normalized.append(key)
                    break

        return cls._dedupe_keep_order(normalized)

    @classmethod
    def _parse_multi_choice_response(cls, response: str, choices: Dict[str, str]) -> str:
        if not response or not choices:
            return ""

        explicit = cls._strip_prediction_wrappers(response).strip()
        explicit_upper = explicit.upper()
        if explicit_upper in choices:
            return explicit_upper

        candidates: List[Tuple[int, str]] = []
        search_text = f" {response.strip()} "
        for key in choices:
            patterns = [
                rf"\({key}\)",
                rf"\b{key}\b",
                rf"{key}\.",
                rf"{key}\)",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, search_text, flags=re.IGNORECASE):
                    candidates.append((match.start(), key))

        response_lower = response.lower()
        for key, value in choices.items():
            option_text = str(value).strip().lower()
            if option_text and option_text in response_lower:
                candidates.append((response_lower.rfind(option_text), key))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        for key, value in choices.items():
            if explicit.lower() == str(value).strip().lower():
                return key

        return ""

    @classmethod
    def _is_open_correct(cls, gold_answer: Any, pred_candidates: List[str]) -> bool:
        gold_candidates = cls._normalize_open_candidates(cls._parse_answer_field(gold_answer))
        if not gold_candidates or not pred_candidates:
            return False

        for pred in pred_candidates:
            for gold in gold_candidates:
                if pred == gold:
                    return True
                try:
                    if abs(float(pred) - float(gold)) < 1e-6:
                        return True
                except Exception:
                    continue
        return False

    def extract_answer(self, prediction: str) -> str:
        if prediction is None:
            return ""

        stripped = self._strip_prediction_wrappers(prediction).strip()
        if not stripped:
            return ""

        if re.fullmatch(r"[A-J]", stripped, flags=re.IGNORECASE):
            return stripped.upper()

        bracketed = re.findall(r"\(([A-J])\)", stripped, flags=re.IGNORECASE)
        if bracketed:
            return bracketed[-1].upper()

        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", stripped, flags=re.IGNORECASE)
        if numbers:
            return numbers[-1]

        return stripped.strip()

    @classmethod
    def create_query(cls, problem: Dict, shot_type: str = "solution", use_caption: bool = False, use_ocr: bool = False):
        del shot_type, use_caption, use_ocr

        question = cls._normalize_question_text(problem.get("question", ""))
        options = cls._parse_options(problem.get("options"))
        question_type = cls._question_type(problem)

        parts = [f"Question: {question}"]
        if question_type == "multiple-choice" and options:
            option_lines = ["Choices:"]
            for idx, option in enumerate(options):
                option_lines.append(f"({chr(ord('A') + idx)}) {option}")
            parts.append("\n".join(option_lines))
            parts.append("Hint: Answer with the option letter only, for example A, B, C, or D.")
        else:
            parts.append("Hint: Answer the question using a short final response.")

        return "\n".join(part for part in parts if part).strip()

    def _create_vlm_messages(self, question: str, state: DatasetState, item: Dict = None):
        messages = []
        sys_prompt = self.cot_system_prompt if "TEACHER" in state.value else self.direct_answer_system_prompt
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})

        images = self._collect_images(item) if item is not None else []
        user_content = []
        for image in images:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        if "TEACHER" in state.value:
            if item is None:
                raise ValueError("Teacher message requires item with self_cot.")
            record = self._self_cot_data.get(str(item.get("id"))) if self._self_cot_data else None
            cot = record.get("self_cot", "") if record else item.get("self_cot", "")
            if not isinstance(cot, str) or not cot.strip():
                raise ValueError(f"Missing self_cot for id={item.get('id')}")
            answer_text = self._canonical_answer_text(
                record.get("gt_numerical", item.get("answer")) if record else item.get("answer")
            )
            teacher_msg = f"{cot.strip()}\n#### {answer_text}"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": teacher_msg}]})

        return messages

    def _parse_prediction(self, item: Dict, prediction: str) -> Union[str, List[str]]:
        if self._question_type(item) == "multiple-choice":
            options = self._parse_options(item.get("options"))
            choices = self._list_to_dict(options)
            return self._parse_multi_choice_response(prediction, choices)
        return self._parse_open_response(prediction)

    def _is_correct(self, item: Dict, extracted_answer: str, raw_prediction: Optional[str] = None) -> bool:
        prediction_text = raw_prediction if raw_prediction is not None else extracted_answer

        if self._question_type(item) == "multiple-choice":
            options = self._parse_options(item.get("options"))
            choices = self._list_to_dict(options)
            gold_answers = self._normalize_gold_choices(item.get("answer"), choices)
            pred = self._parse_multi_choice_response(prediction_text, choices)
            if not pred and extracted_answer:
                pred = self._parse_multi_choice_response(extracted_answer, choices)
            return bool(pred) and pred in gold_answers

        pred_candidates = self._parse_open_response(prediction_text)
        if not pred_candidates and extracted_answer:
            pred_candidates = self._parse_open_response(extracted_answer)
        return self._is_open_correct(item.get("answer"), pred_candidates)

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        teacher_prompts, teacher_images, teacher_ans_token_ids = [], [], []
        student_id_list, student_images, student_answer_lengths = [], [], []

        for item in batch_data:
            question = self.create_query(item)
            images = self._collect_images(item)
            record = self._self_cot_data.get(str(item.get("id"))) if self._self_cot_data else None
            final_answer = self._canonical_answer_text(
                record.get("gt_numerical", item.get("answer")) if record else item.get("answer")
            )

            if self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT:
                teacher_state = DatasetState.TRAIN_TEACHER_SELF_COT
            elif self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q:
                teacher_state = DatasetState.TRAIN_TEACHER
            else:
                teacher_state = (
                    DatasetState.TRAIN_TEACHER_SELF_COT
                    if "SELF_COT" in self.dataset_state.value
                    else DatasetState.TRAIN_TEACHER
                )

            teacher_messages = self._create_vlm_messages(question, teacher_state, item=item)
            teacher_text = self.model_processor.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            teacher_prompts.append(teacher_text)
            teacher_images.append(images)

            answer_ids_batch = self.model_processor(text=final_answer, add_special_tokens=False).input_ids
            answer_ids = answer_ids_batch[0].tolist() if hasattr(answer_ids_batch, "tolist") else answer_ids_batch[0]
            if len(answer_ids) == 0:
                raise ValueError(f"Empty answer tokenization for id={item.get('id')}, answer={final_answer!r}")
            teacher_ans_token_ids.append(answer_ids)

            if self.dataset_state in [DatasetState.TRAIN_STUDENT_DIRECT_Q, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                student_state = self.dataset_state
            else:
                student_state = (
                    DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT
                    if "SELF_COT" in self.dataset_state.value
                    else DatasetState.TRAIN_STUDENT_DIRECT_Q
                )

            student_messages = self._create_vlm_messages(question, student_state, item=item)
            student_text = self.model_processor.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            student_inputs = self.model_processor(text=[student_text], images=[images], return_tensors="pt")
            student_prompt_ids = student_inputs.input_ids[0].tolist()

            eos_id = [self.model_processor.tokenizer.eos_token_id]
            full_student_ids = [int(x) for x in (student_prompt_ids + answer_ids + eos_id)]
            student_id_list.append(torch.tensor(full_student_ids, dtype=torch.long))
            student_images.append(images)
            student_answer_lengths.append(len(answer_ids))

        teacher_batch = self.model_processor(
            text=teacher_prompts,
            images=teacher_images,
            return_tensors="pt",
            padding=True,
        )
        teacher_answer_mask = torch.zeros_like(teacher_batch.input_ids, dtype=torch.bool)
        for i, answer_seq in enumerate(teacher_ans_token_ids):
            full_list = teacher_batch.input_ids[i].tolist()
            matched = False
            for j in range(len(full_list) - len(answer_seq), -1, -1):
                if full_list[j:j + len(answer_seq)] == answer_seq:
                    teacher_answer_mask[i, j:j + len(answer_seq)] = True
                    matched = True
                    break
            if not matched:
                answer_text = self._canonical_answer_text(batch_data[i].get("answer"))
                err_msg = (
                    f"Failed to locate teacher answer span for batch sample {i}, "
                    f"id={batch_data[i].get('id')}, answer={answer_text!r}"
                )
                if self.skip_overlength_samples:
                    print(f"Warning: {err_msg}. Skipping/resampling MMMU sample.")
                    if len(batch_data) == 1:
                        return self._resample_single_train_sample(max_seq_len, err_msg)

                    filtered_batch = [
                        sample for idx, sample in enumerate(batch_data) if idx != i
                    ]
                    if filtered_batch:
                        return self.collate_fn_for_train(filtered_batch, max_seq_len)
                    return self._resample_single_train_sample(max_seq_len, err_msg)
                raise ValueError(err_msg)

        student_padded = torch.nn.utils.rnn.pad_sequence(
            student_id_list,
            batch_first=True,
            padding_value=self.model_processor.tokenizer.pad_token_id,
            padding_side="left",
        )

        if student_padded.shape[1] > max_seq_len:
            if not self.skip_overlength_samples:
                raise ValueError(
                    f"MMMU student sequence length {student_padded.shape[1]} exceeds max_seq_len={max_seq_len}. "
                    "Increase data.max_seq_len for extraction to preserve multimodal alignment."
                )

            # Skip overlength samples instead of crashing the whole run.
            # For current MMMU training we use batch_size=1, so we can safely resample.
            sample_id = batch_data[0].get("id") if len(batch_data) == 1 else "mixed-batch"
            print(
                f"Warning: skip MMMU sample id={sample_id} because student sequence length "
                f"{student_padded.shape[1]} exceeds max_seq_len={max_seq_len}."
            )
            if len(batch_data) == 1:
                return self._resample_single_train_sample(
                    max_seq_len,
                    (
                        f"MMMU student sequence length {student_padded.shape[1]} "
                        f"exceeds max_seq_len={max_seq_len}"
                    ),
                )

            # For batch_size > 1, drop the current longest sequence sample and retry,
            # mirroring the skip-bad-sample behavior used in extraction code.
            longest_idx = max(
                range(len(student_id_list)),
                key=lambda idx: int(student_id_list[idx].shape[0]),
            )
            filtered_batch = [
                sample for idx, sample in enumerate(batch_data) if idx != longest_idx
            ]
            if filtered_batch:
                return self.collate_fn_for_train(filtered_batch, max_seq_len)

            raise ValueError(
                f"MMMU student sequence length {student_padded.shape[1]} exceeds max_seq_len={max_seq_len}. "
                "Use batch_size=1 for resample-based skipping or increase data.max_seq_len."
            )

        student_labels = torch.full_like(student_padded, -100)
        student_answer_mask = torch.zeros_like(student_padded, dtype=torch.bool)
        for i, answer_len in enumerate(student_answer_lengths):
            answer_start_idx = -answer_len - 1
            answer_end_idx = -1
            student_labels[i, answer_start_idx:answer_end_idx] = student_padded[i, answer_start_idx:answer_end_idx]
            student_answer_mask[i, answer_start_idx:answer_end_idx] = True

        student_pixels = self.model_processor(
            text=[""] * len(student_images),
            images=student_images,
            return_tensors="pt",
        )

        return {
            "prefix_inputs": teacher_batch,
            "teacher_answer_mask": teacher_answer_mask,
            "student_inputs": {
                "input_ids": student_padded,
                "pixel_values": student_pixels.pixel_values,
                "image_grid_thw": student_pixels.image_grid_thw,
                "attention_mask": student_padded.ne(self.model_processor.tokenizer.pad_token_id).long(),
            },
            "student_labels": student_labels,
            "student_answer_mask": student_answer_mask,
        }

    def train_dataloader(self, model: Any, batch_size: int) -> DataLoader:
        del model
        mode = getattr(self.data_cfg, "training_mode", "TRAIN_STUDENT_DIRECT_Q")
        self.dataset_state = DatasetState(mode)
        return DataLoader(
            self._support_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate_fn_for_train(
                batch,
                getattr(self.data_cfg, "max_seq_len", 4096),
            ),
        )

    def eval(self, eval_cfg: DictConfig, model: Any) -> Tuple[List[Dict], Dict]:
        eval_mode = getattr(eval_cfg, "eval_mode", DatasetState.EVAL_BASELINE.value)
        if eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q
        elif eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT
        else:
            self.dataset_state = DatasetState.EVAL_BASELINE

        loader = DataLoader(self._query_set, batch_size=eval_cfg.batch_size, shuffle=False, collate_fn=lambda x: x)
        records: List[Dict] = []
        correct = 0

        for batch in tqdm(loader, desc="MM Eval MMMU"):
            prompts: List[str] = []
            images: List[List[Image.Image]] = []
            queries: List[str] = []

            for item in batch:
                query = self.create_query(item)
                sample_images = self._collect_images(item)
                if self.model_name and "llava-onevision" in self.model_name.lower():
                    prompt = query
                else:
                    messages = self._create_vlm_messages(query, self.dataset_state, item=item)
                    prompt = self.model_processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                prompts.append(prompt)
                images.append(sample_images)
                queries.append(query)

            outputs = model.generate(prompts, images=images, **eval_cfg.generation_args)

            for idx, prediction in enumerate(outputs):
                item = batch[idx]
                parsed_pred = self._parse_prediction(item, prediction)
                is_correct = self._is_correct(item, prediction)
                if is_correct:
                    correct += 1

                records.append(
                    {
                        "id": item.get("id"),
                        "pid": item.get("id"),
                        "subject": item.get("subject", ""),
                        "question_type": item.get("question_type", ""),
                        "question": item.get("question", ""),
                        "options": self._parse_options(item.get("options")),
                        "answer": item.get("answer", ""),
                        "query": queries[idx],
                        "response": prediction,
                        "raw_extraction": self.extract_answer(prediction),
                        "parsed_pred": parsed_pred,
                        "is_correct": is_correct,
                    }
                )

        accuracy = correct / len(records) if records else 0.0
        return records, {"accuracy": accuracy}
