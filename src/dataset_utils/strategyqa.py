import re
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset, Dataset as HFDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import evaluate
import os # Import os module
from tqdm import tqdm # Import tqdm
import random # Import random
import enum
import torch
import json
from collections import Counter

from .interface import DatasetBase
from testbed.data import prepare_dataloader, prepare_input
from testbed.models.model_base import ModelBase


class DatasetState(enum.Enum):
    TRAIN_TEACHER = "TRAIN_TEACHER"
    TRAIN_STUDENT_ONESHOT = "TRAIN_STUDENT_ONESHOT"
    TRAIN_STUDENT_DIRECT_Q = "TRAIN_STUDENT_DIRECT_Q"
    # New Self-CoT training modes
    TRAIN_TEACHER_SELF_COT = "TRAIN_TEACHER_SELF_COT"  # Teacher uses Self-CoT instead of GT-CoT
    TRAIN_STUDENT_ONESHOT_SELF_COT = "TRAIN_STUDENT_ONESHOT_SELF_COT"  # Student with Self-CoT teacher
    TRAIN_STUDENT_DIRECT_Q_SELF_COT = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"  # Student with Self-CoT teacher
    EVAL_BASELINE = "EVAL_BASELINE" # Existing eval mode
    EVAL_WITH_COT_VECTOR_ONESHOT = "EVAL_WITH_COT_VECTOR_ONESHOT"
    EVAL_WITH_COT_VECTOR_DIRECT_Q = "EVAL_WITH_COT_VECTOR_DIRECT_Q"

# only for StrategyQA
class StrategyQADataset(DatasetBase):
    support_datasets = ["strategyqa"]

    def _tokenizer(self):
        if hasattr(self.model_processor, "tokenizer"):
            return self.model_processor.tokenizer
        return self.model_processor

    def __init__(self, data_cfg: DictConfig, model_processor: Any = None, model_name: Optional[str] = None) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        local_path = getattr(data_cfg, "source", "/data/share/strategy_qa/")
        self._dataset = load_dataset(
            "parquet", 
            data_files={
                "train": os.path.join(local_path, "train-00000-of-00001-506370352f622815.parquet"),
                "test": os.path.join(local_path, "test-00000-of-00001-bae602f3ee37f4ca.parquet"),
            }
        )
        indices = list(range(len(self._dataset["test"])))
        import random
        random.seed(42)
        random.shuffle(indices)
        query_limit = getattr(data_cfg, "num_query_samples", None)
        if query_limit is None or int(query_limit) <= 0:
            query_limit = len(indices)
        query_limit = min(int(query_limit), len(indices))
        self._query_set = self._dataset["test"].select(indices[:query_limit])
        self._support_set = self._dataset["train"]
        self._validation_set = self._query_set.select(range(min(16, len(self._query_set))))
        self.data_cfg = data_cfg  # Store data_cfg
        self.dataset_state = DatasetState.EVAL_BASELINE  # Default state


        # Load Self-CoT data if specified
        self._self_cot_data = None
        self._filtered_support_set = None
        if hasattr(data_cfg, "use_self_cot") and data_cfg.use_self_cot:
            self._load_self_cot_data(data_cfg)

        # Add sampling logic for training data based on num_query_samples
        if hasattr(data_cfg, "num_query_samples") and data_cfg.num_query_samples > 0:
            num_samples_to_take = min(data_cfg.num_query_samples, len(self._support_set))
            self._support_set = self._support_set.select(
                random.sample(range(len(self._support_set)), num_samples_to_take))
            print(
                f"StrategyQA Dataset initialized: Training set limited to {len(self._support_set)} samples by num_query_samples.")

        # Qwen2.5-Math-7B-Instruct uses a chat template with system/user roles.
        # The prompt for direct answer.
        self.direct_answer_system_prompt = "You are a helpful and precise assistant for solving math problems. Put your final answer within \\boxed{}."
        # The prompt for Chain of Thought.
        self.cot_system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    def _load_self_cot_data(self, data_cfg: DictConfig):
        """Load Self-CoT data and filter support set to only include correct samples."""
        self_cot_path = getattr(data_cfg, "self_cot_path", None)
        if not self_cot_path:
            raise ValueError("use_self_cot is True but self_cot_path is not specified")
        
        if not os.path.exists(self_cot_path):
            raise FileNotFoundError(f"Self-CoT data file not found: {self_cot_path}")

        # Load Self-CoT data
        self._self_cot_data = {}
        with open(self_cot_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                question = data['question']
                data['gt_answer'] = data['gt_numerical']
                data['cot_content'] = data['self_cot']
                self._self_cot_data[question] = data


        # Determine if the loaded file is already 'correct_only'
        is_correct_only_file = "correct_only" in os.path.basename(self_cot_path)

        # Filter support set to only include correct samples (if not already a correct-only file)
        filtered_samples = []
        for sample in self._support_set:
            question = sample['question']
            if question in self._self_cot_data:
                self_cot_info = self._self_cot_data[question]
                if is_correct_only_file or self_cot_info['is_correct']:
                    # Add Self-CoT data to the sample
                    sample_with_self_cot = dict(sample)
                    sample_with_self_cot['self_cot'] = self_cot_info['self_cot']
                    sample_with_self_cot['cot_content'] = self_cot_info['self_cot']
                    sample_with_self_cot['gt_answer'] = self_cot_info['gt_answer']
                    filtered_samples.append(sample_with_self_cot)

        self._filtered_support_set = filtered_samples
        print(
            f"Loaded {len(self._self_cot_data)} Self-CoT samples, filtered to {len(self._filtered_support_set)} correct samples")

        if getattr(data_cfg, "balance_labels", False):
            balance_strategy = getattr(data_cfg, "balance_strategy", "oversample_minority")
            balance_seed = int(getattr(data_cfg, "balance_seed", 42))
            self._filtered_support_set = self._rebalance_filtered_support_set(
                self._filtered_support_set,
                strategy=balance_strategy,
                seed=balance_seed,
            )

        # Update support set to use filtered data
        if self._filtered_support_set:
            self._support_set = HFDataset.from_list(self._filtered_support_set)

    def _rebalance_filtered_support_set(
        self,
        samples: List[Dict[str, Any]],
        strategy: str = "oversample_minority",
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        if not samples:
            print("StrategyQA balance skipped: no filtered samples available.")
            return samples

        grouped: Dict[str, List[Dict[str, Any]]] = {"True": [], "False": []}
        for sample in samples:
            label = self._normalize_binary_answer(sample.get("gt_answer"))
            if label in grouped:
                grouped[label].append(sample)

        counts = {label: len(items) for label, items in grouped.items() if items}
        print(f"StrategyQA balance before: {counts}")

        if len(counts) < 2:
            print("StrategyQA balance skipped: only one label present after filtering.")
            return samples

        rng = random.Random(seed)
        if strategy == "oversample_minority":
            target_count = max(len(grouped["True"]), len(grouped["False"]))
            rebalanced: List[Dict[str, Any]] = []
            for label in ("True", "False"):
                items = list(grouped[label])
                if not items:
                    continue
                if len(items) < target_count:
                    extra = [dict(rng.choice(items)) for _ in range(target_count - len(items))]
                    items.extend(extra)
                rebalanced.extend(items)
        elif strategy == "downsample_majority":
            target_count = min(len(grouped["True"]), len(grouped["False"]))
            rebalanced = []
            for label in ("True", "False"):
                items = list(grouped[label])
                if not items:
                    continue
                if len(items) > target_count:
                    items = rng.sample(items, target_count)
                rebalanced.extend(items)
        else:
            raise ValueError(f"Unsupported StrategyQA balance strategy: {strategy}")

        rng.shuffle(rebalanced)
        after_counts = Counter(
            self._normalize_binary_answer(sample.get("gt_answer")) for sample in rebalanced
        )
        print(f"StrategyQA balance after: {dict(after_counts)}")
        return rebalanced



    @staticmethod
    def metric_key() -> str:
        return "accuracy"

    @property
    def instruction(self) -> str:
        # This will be determined by the specific evaluation mode (direct or CoT)
        # and will be passed to prepare_input in the eval method.
        return "" 

    @property
    def num_role_in_round(self) -> int:
        return 2  # user and assistant roles

    def _normalize_binary_answer(self, answer_text: Optional[str]) -> Optional[str]:
        if answer_text is None:
            return None

        text = str(answer_text).strip()
        if not text:
            return None

        def _map_binary_label(label: str) -> Optional[str]:
            normalized = str(label).strip().lower()
            if normalized in {"true", "yes"}:
                return "True"
            if normalized in {"false", "no"}:
                return "False"
            return None

        explicit_patterns = [
            r"\\boxed\{\s*(?:\\text\{)?\s*(true|false|yes|no)\s*(?:\})?\s*\}",
            r"boxed\{\s*(?:text\{)?\s*(true|false|yes|no)\s*(?:\})?\s*\}",
            r"answer is\s*\(?\s*(true|false|yes|no)\s*\)?",
            r"(?:final answer|correct answer|the answer)\D+(true|false|yes|no)\b",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                mapped = _map_binary_label(match.group(1))
                if mapped is not None:
                    return mapped

        exact_match = re.fullmatch(r"(true|false|yes|no)", text, re.IGNORECASE)
        if exact_match:
            return _map_binary_label(exact_match.group(1))

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-3:]):
            line_match = re.fullmatch(r"(true|false|yes|no)", line, re.IGNORECASE)
            if line_match:
                return _map_binary_label(line_match.group(1))

        tail_text = text[-160:]
        tail_line_matches = re.findall(
            r"(?:therefore|thus|so|hence|overall|in conclusion|final answer is)\D*(true|false|yes|no)\b",
            tail_text,
            flags=re.IGNORECASE,
        )
        if tail_line_matches:
            return _map_binary_label(tail_line_matches[-1])

        final_mentions = re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)
        if final_mentions:
            return final_mentions[-1].capitalize()

        yes_no_tail_mentions = re.findall(r"\b(yes|no)\b", tail_text, flags=re.IGNORECASE)
        if yes_no_tail_mentions:
            return _map_binary_label(yes_no_tail_mentions[-1])

        return None

    def extract_answer(self, prediction: str) -> str:
        return self._normalize_binary_answer(prediction)

    def _extract_final_answer_from_full_cot(self, full_cot_answer: str) -> str:
        # Extract just the terminal True/False answer from a full rationale.
        return self.extract_answer(full_cot_answer)

    def _extract_cot_only(self, full_cot_answer: str) -> str:
        final_answer = self.extract_answer(full_cot_answer)
        if final_answer:
            matches = list(re.finditer(re.escape(final_answer), full_cot_answer, re.IGNORECASE))
            if matches:
                return full_cot_answer[:matches[-1].start()].strip()
        return full_cot_answer.strip()

    def _format_single_example(self, example: Dict, use_cot: bool, model_name: str, for_teacher: bool = False) -> str:
        question = example["question"]
        full_cot_answer = example["answer"] # This contains the full CoT and "#### final_answer"

        if for_teacher:
            # Teacher receives Q + GT_CoT (truncated)
            cot_part = self._extract_cot_only(full_cot_answer)
            return f"{question}\n{cot_part}"
        
        if use_cot:
            # For CoT mode (e.g., baseline eval, or if student was trained for full CoT generation),
            # use the full ground truth answer, including CoT if present.
            return f"{question}\n{full_cot_answer}"
        else:
            # For direct answer mode (e.g., student 1-shot example, or baseline eval direct answer),
            # extract only the final answer and box it.
            extracted_answer = self._extract_final_answer_from_full_cot(full_cot_answer)
            return f"{question}\n\\boxed{{{extracted_answer}}}"

    def preprocess_options(self, options: List[str]) -> str:
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # 将选项列表转换为格式化字符串
        formatted_options = []
        for i, option in enumerate(options):
            if i < len(option_letters):
                formatted_options.append(f"{option_letters[i]}. {option}")
            else:
                # 如果选项超过字母数量，使用数字
                formatted_options.append(f"{i + 1}. {option}")

        # 用逗号连接所有选项
        str_options = "; ".join(formatted_options)
        return str_options

    def _resolve_eval_state(self, eval_mode: Optional[str]) -> DatasetState:
        if eval_mode is None:
            return DatasetState.EVAL_BASELINE

        normalized_mode = str(eval_mode).upper()
        if normalized_mode == DatasetState.EVAL_BASELINE.value:
            return DatasetState.EVAL_BASELINE
        if normalized_mode == DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT.value:
            return DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT
        if normalized_mode == DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value:
            return DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q

        raise ValueError(
            f"Unsupported StrategyQA eval_mode: {eval_mode}. "
            f"Supported modes are {DatasetState.EVAL_BASELINE.value}, "
            f"{DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT.value}, and "
            f"{DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value}."
        )

    def _format_eval_example_answer(self, example: Dict, use_cot: bool) -> str:
        if use_cot and example.get("cot_content"):
            return str(example["cot_content"])

        answer = example.get("gt_answer", example.get("answer"))
        normalized = self._normalize_binary_answer(answer)
        if normalized is not None:
            return normalized
        if answer is not None:
            return str(answer)

        raise KeyError("Few-shot example is missing an answer field.")


    def _create_qwen_chat_template(self, question: str, use_cot: bool,
                                   few_shot_examples: List[Dict] = None, dataset_state: DatasetState = DatasetState.EVAL_BASELINE) -> List[Dict]:
        messages = []

        system_prompt = ""
        # Determine system prompt based on whether final query expects CoT
        if dataset_state in [DatasetState.EVAL_BASELINE, DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT, DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q]:
            if use_cot:
                # system_prompt = "The following are multiple choice questions (with answers) about {$}. " \
                #                 "Think step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice."
                system_prompt = "You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \\boxed{}." \
                                "Your final output should be only True or False."
            else:
                system_prompt = ("You are a helpful and precise assistant for solving problems. Put your answer within \\boxed{}."
                                 "Your final output should be only True or False.")
        elif dataset_state in [DatasetState.TRAIN_TEACHER, DatasetState.TRAIN_TEACHER_SELF_COT]:
            # Teacher model: expects full CoT and final answer from the dataset.
            # The system prompt should indicate a step-by-step reasoning with a boxed answer.
            # system_prompt = "The following are multiple choice questions (with answers) about {$}. " \
            #                 "Think step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice."
            system_prompt = "You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \\boxed{}." \
                            "Your final output should be only True or False."
        elif dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_DIRECT_Q, 
                              DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
            # Student model: for training, we now ask for direct numerical answer, no \boxed{} for simplicity.
            # The CoT effect comes from the shift vector.
            # system_prompt = "You are a helpful and precise assistant for solving math problems. Output the final numerical answer."
            system_prompt = ("You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \\boxed{}."
                             "Your final output should be only True or False.")
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add few-shot examples for student model (TRAIN_STUDENT_ONESHOT) or evaluation (EVAL_BASELINE etc.)
        if few_shot_examples and (dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.EVAL_BASELINE, 
                                                    DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT,
                                                    DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT]):
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["question"]})
                # For student few-shot training, we want the final numerical answer ONLY, no \boxed{}.
                # For eval, if EVAL_BASELINE or EVAL_WITH_COT_VECTOR_ONESHOT, we still use \boxed{} as that's the expected output format for final answer.
                formatted_example_answer = self._format_eval_example_answer(example, use_cot)
                if dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT]:
                    messages.append({"role": "assistant",
                                     "content": f"{formatted_example_answer}"})
                else:
                    messages.append({"role": "assistant", "content": f"\\boxed{{{formatted_example_answer}}}"})

        # Main query part
        if dataset_state in [DatasetState.TRAIN_TEACHER, DatasetState.TRAIN_TEACHER_SELF_COT]:
            # For TRAIN_TEACHER, `few_shot_examples[0]["answer"]` contains the *full* GT CoT + final answer for the current query.
            # We want the teacher's input to be Q (user) + Full CoT + Final Answer (assistant).
            messages.append({"role": "user", "content": question})
            if few_shot_examples and len(few_shot_examples) > 0:
                example = few_shot_examples[0]
                example_cot_sequence = example['cot_content']
                if dataset_state == DatasetState.TRAIN_TEACHER_SELF_COT and 'self_cot' in example:
                    # Use Self-CoT + final answer for TRAIN_TEACHER_SELF_COT
                    # The answer is (I).
                    self_cot_with_answer = f"{example['self_cot']}" #  The answer is ({example['gt_answer']}).
                    messages.append({"role": "assistant", "content": self_cot_with_answer})
                else:
                    # Fallback to GT-CoT for TRAIN_TEACHER or if Self-CoT not available
                    messages.append({"role": "assistant",
                                     "content": example_cot_sequence})
            else:
                # This case should ideally not happen for TRAIN_TEACHER as `item` is passed as few_shot_examples
                # But as a fallback, if no examples are somehow passed, just add the question.
                pass 
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        def _to_id_list(batch_ids):
            if hasattr(batch_ids, "tolist"):
                batch_ids = batch_ids.tolist()
            if isinstance(batch_ids, list) and batch_ids and isinstance(batch_ids[0], list):
                batch_ids = batch_ids[0]
            return [int(token_id) for token_id in batch_ids]

        teacher_state = self.dataset_state
        if self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
            teacher_state = DatasetState.TRAIN_TEACHER_SELF_COT
        elif self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_DIRECT_Q]:
            teacher_state = DatasetState.TRAIN_TEACHER

        teacher_prompts = []
        teacher_answer_token_ids = []
        teacher_plain_answer_token_ids = []
        student_id_list = []
        student_answer_lengths = []
        cot_texts = []

        for item in batch_data:
            question = item["question"]
            cot_text = item["cot_content"]
            cot_texts.append(cot_text)

            final_answer = self._normalize_binary_answer(item.get("gt_answer"))
            if final_answer is None:
                final_answer = self._extract_final_answer_from_full_cot(cot_text)
            if final_answer is None:
                raise ValueError(
                    f"Failed to resolve StrategyQA final answer for question={question!r}"
                )

            teacher_prompt_messages = self._create_qwen_chat_template(
                question=question,
                use_cot=True,
                few_shot_examples=[item],
                dataset_state=teacher_state,
            )
            teacher_prompt_text = self.model_processor.apply_chat_template(
                teacher_prompt_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            teacher_prompts.append(teacher_prompt_text)

            answer_ids = _to_id_list(
                self.model_processor(text=final_answer, add_special_tokens=False).input_ids
            )
            if not answer_ids:
                raise ValueError(
                    f"Empty answer tokenization for StrategyQA answer={final_answer!r}"
                )
            teacher_plain_answer_token_ids.append(answer_ids)

            teacher_ctx_ids = _to_id_list(
                self.model_processor(text=" " + final_answer, add_special_tokens=False).input_ids
            )
            teacher_answer_token_ids.append(
                teacher_ctx_ids if len(teacher_ctx_ids) > 0 else answer_ids
            )

            if self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT]:
                if len(self._support_set) < 1:
                    raise ValueError("Not enough examples in training set for 1-shot learning.")
                one_shot_example = random.sample(list(self._support_set), 1)[0]
                student_prompt_messages = self._create_qwen_chat_template(
                    question=question,
                    use_cot=False,
                    few_shot_examples=[one_shot_example],
                    dataset_state=self.dataset_state,
                )
            elif self.dataset_state in [DatasetState.TRAIN_STUDENT_DIRECT_Q, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                student_prompt_messages = self._create_qwen_chat_template(
                    question=question,
                    use_cot=False,
                    few_shot_examples=None,
                    dataset_state=self.dataset_state,
                )
            else:
                raise ValueError(f"Invalid dataset state for training collate_fn: {self.dataset_state}")

            student_prompt_text = self.model_processor.apply_chat_template(
                student_prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            student_prompt_inputs = self.model_processor(
                text=[student_prompt_text],
                return_tensors="pt",
            )
            student_prompt_ids = _to_id_list(student_prompt_inputs.input_ids[0])

            student_answer_ids = answer_ids
            full_student_text = student_prompt_text + final_answer
            full_student_inputs = self.model_processor(
                text=[full_student_text],
                return_tensors="pt",
            )
            full_student_ids = _to_id_list(full_student_inputs.input_ids[0])
            if full_student_ids[: len(student_prompt_ids)] == student_prompt_ids:
                student_answer_ids = full_student_ids[len(student_prompt_ids) :]

            eos_id = [self._tokenizer().eos_token_id]
            full_student_sequence = [
                int(token_id)
                for token_id in (student_prompt_ids + student_answer_ids + eos_id)
            ]
            student_id_list.append(torch.tensor(full_student_sequence, dtype=torch.long))
            student_answer_lengths.append(len(student_answer_ids))

        teacher_inputs = self.model_processor(
            text=teacher_prompts,
            return_tensors="pt",
            padding=True,
        )

        batch_teacher_cot_mask = torch.zeros_like(teacher_inputs["input_ids"], dtype=torch.bool)
        for idx, cot_text in enumerate(cot_texts):
            if not cot_text:
                continue
            cot_ids = _to_id_list(
                self.model_processor(text=cot_text, add_special_tokens=False).input_ids
            )
            if not cot_ids:
                continue

            full_ids = teacher_inputs["input_ids"][idx].tolist()
            start = -1
            for pos in range(len(full_ids) - len(cot_ids) + 1):
                if full_ids[pos: pos + len(cot_ids)] == cot_ids:
                    start = pos
                    break
            if start != -1:
                batch_teacher_cot_mask[idx, start:] = True

        batch_teacher_answer_mask = torch.zeros_like(teacher_inputs["input_ids"], dtype=torch.bool)
        for idx, answer_seq in enumerate(teacher_answer_token_ids):
            full_ids = teacher_inputs["input_ids"][idx].tolist()
            matched = False
            for start in range(len(full_ids) - len(answer_seq), -1, -1):
                if full_ids[start: start + len(answer_seq)] == answer_seq:
                    batch_teacher_answer_mask[idx, start: start + len(answer_seq)] = True
                    matched = True
                    break

            if not matched and teacher_plain_answer_token_ids[idx] != answer_seq:
                plain_seq = teacher_plain_answer_token_ids[idx]
                for start in range(len(full_ids) - len(plain_seq), -1, -1):
                    if full_ids[start: start + len(plain_seq)] == plain_seq:
                        batch_teacher_answer_mask[idx, start: start + len(plain_seq)] = True
                        matched = True
                        break

            if not matched:
                sample_answer = self._normalize_binary_answer(batch_data[idx].get("gt_answer"))
                if sample_answer is None:
                    sample_answer = self._normalize_binary_answer(batch_data[idx].get("answer"))
                raise ValueError(
                    f"Failed to locate teacher answer span for batch sample {idx}, "
                    f"question={batch_data[idx].get('question', '')!r}, answer={sample_answer!r}"
                )

        student_inputs_padded = torch.nn.utils.rnn.pad_sequence(
            student_id_list,
            batch_first=True,
            padding_value=self._tokenizer().pad_token_id,
            padding_side="left",
        )
        if student_inputs_padded.shape[1] > max_seq_len:
            student_inputs_padded = student_inputs_padded[:, -max_seq_len:]

        student_labels = torch.full_like(student_inputs_padded, -100)
        student_answer_mask = torch.zeros_like(student_inputs_padded, dtype=torch.bool)
        for idx, answer_len in enumerate(student_answer_lengths):
            answer_start_idx = -answer_len - 1
            answer_end_idx = -1
            student_labels[idx, answer_start_idx:answer_end_idx] = student_inputs_padded[
                idx, answer_start_idx:answer_end_idx
            ]
            student_answer_mask[idx, answer_start_idx:answer_end_idx] = True

        student_inputs = {
            "input_ids": student_inputs_padded,
            "attention_mask": student_inputs_padded.ne(self._tokenizer().pad_token_id).long(),
        }

        return {
            "prefix_inputs": teacher_inputs,
            "teacher_answer_mask": batch_teacher_answer_mask,
            "teacher_cot_mask": batch_teacher_cot_mask,
            "student_inputs": student_inputs,
            "student_labels": student_labels,
            "student_answer_mask": student_answer_mask,
            "images": [[] for _ in batch_data],
        }

    def train_dataloader(self, model: ModelBase, batch_size: int, distributed: bool = True) -> DataLoader:
        # Determine training mode from data_cfg.training_mode
        training_mode = getattr(self.data_cfg, "training_mode", "TRAIN_STUDENT_DIRECT_Q")
        self.dataset_state = DatasetState(training_mode) # Set the dataset state for prompt formatting

        # The data_cfg.model_processor is needed for apply_chat_template in collate_fn_for_train
        # So, we pass the model.processor (tokenizer) into data_cfg here.
        # This is a bit of a hack, but hydra instantiates DataModule first.
        self.max_seq_len = getattr(self.data_cfg, "max_seq_len", 2048) # Default max_seq_len

        return DataLoader(
            self._support_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate_fn_for_train(batch, self.max_seq_len) # Pass max_seq_len
        )

    def collate_fn_for_validation(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        sampled_few_shot_examples = []
        batch_messages = []
        ground_truths = []
        for idx, item in enumerate(batch_data):
            # For eval, the few-shot examples are passed to the chat template.
            # The `use_cot` for template creation depends on `eval_cfg.use_cot` for baseline
            # or the `eval_mode` for cot vector eval.
            question = item["question"]
            ground_truth = item["answer"]

            messages = self._create_qwen_chat_template(question=question, use_cot=False,
                                                       few_shot_examples=sampled_few_shot_examples,
                                                       dataset_state=self.dataset_state)
            batch_messages.append(messages)
            ground_truths.append(str(ground_truth))

        return{
            "batch_messages": batch_messages,
            "ground_truths": ground_truths,
        }


    def validation_dataloader(self, batch_size: int) -> DataLoader:
        # For validation, we use the test set and assume a direct Q student model evaluation for simplicity in training.
        # The actual evaluation for trained checkpoints will be handled by eval.py.
        self.dataset_state = DatasetState.TRAIN_STUDENT_DIRECT_Q # Validate using direct Q for student
        self.max_seq_len = getattr(self.data_cfg, "max_seq_len", 2048) # Default max_seq_len

        return DataLoader(
            self._validation_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: self.collate_fn_for_validation(batch, self.max_seq_len) # Pass max_seq_len
        )


    def eval(
        self,
        eval_cfg: DictConfig,
        model: ModelBase,
    ) -> Tuple[List[Dict], Dict]:
        generation_args = {k: v for k, v in eval_cfg.generation_args.items()}
        batch_size = eval_cfg.batch_size
        use_cot = bool(getattr(eval_cfg, "use_cot", False))
        num_shot = eval_cfg.data.num_shot  # Get num_shot from config

        self.dataset_state = self._resolve_eval_state(getattr(eval_cfg, "eval_mode", None))

        if self.dataset_state == DatasetState.EVAL_BASELINE and not use_cot:
            direct_answer_cap = int(getattr(eval_cfg, "direct_answer_max_new_tokens", 8))
            generation_args["max_new_tokens"] = min(
                int(generation_args.get("max_new_tokens", direct_answer_cap)),
                direct_answer_cap,
            )

        records = []
        correct_predictions = 0
        total_samples = 0

        # Sample few-shot examples if num_shot > 0 for EVAL_BASELINE or EVAL_WITH_COT_VECTOR_ONESHOT
        sampled_few_shot_examples = []
        if num_shot > 0 and (self.dataset_state == DatasetState.EVAL_BASELINE or self.dataset_state == DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT):
            # Ensure we don't sample more examples than available in the training set
            num_available_examples = len(self._support_set)
            if num_shot > num_available_examples:
                print(f"Warning: Requested {num_shot} shots, but only {num_available_examples} available in training set. Using all available examples.")
                sampled_few_shot_examples = self._support_set[:] # Take all
            else:
                sampled_few_shot_examples = random.sample(list(self._support_set), num_shot)

        # Determine which dataset to use based on eval_cfg.mode
        dataset_to_load = None
        if getattr(eval_cfg, "mode", "eval") == "generate_self_cot":
            dataset_to_load = self._support_set
            print(f"Mode is 'generate_self_cot', loading training set ({len(dataset_to_load)} samples) for Self-CoT generation.")
        else:
            dataset_to_load = self._query_set
            print(f"Mode is 'eval', loading test set ({len(dataset_to_load)} samples) for evaluation.")


        dataloader = DataLoader(
            dataset_to_load,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset_to_load.collate_fn if hasattr(dataset_to_load, 'collate_fn') else None
            # collate_fn=lambda batch: self.collate_fn_for_eval(batch),
        )

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating StrategyQA")):
            questions = batch["question"]
            ground_truth_answers = batch["answer"]
            if hasattr(ground_truth_answers, "tolist"):
                ground_truth_answers = ground_truth_answers.tolist()
            ground_truth_answers = [str(item) for item in ground_truth_answers]

            inputs = []
            for idx, question in enumerate(questions):
                # For eval, the few-shot examples are passed to the chat template.
                # The `use_cot` for template creation depends on `eval_cfg.use_cot` for baseline
                # or the `eval_mode` for cot vector eval.

                messages = self._create_qwen_chat_template(question=question, use_cot=use_cot,
                                                           few_shot_examples=sampled_few_shot_examples,
                                                           dataset_state=self.dataset_state)
                text = model.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs.append(text)

            try:
                predictions = model.generate(
                    inputs, # inputs is a list of strings (prompts)
                    **generation_args
                )
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if predictions is None:
                # Handle OOM or other failures
                print(f"Skipping batch {batch_idx} due to prediction failure (e.g., OOM). ")
                continue

            for i, pred in enumerate(predictions):
                question = questions[i]
                ground_truth = str(ground_truth_answers[i])
                
                extracted_pred = self.extract_answer(pred)
                # extracted_ground_truth = self.extract_answer(ground_truth)

                is_correct = (extracted_pred == ground_truth)
                
                records.append({
                    "question": question,
                    "model_output": pred,
                    "extracted_prediction": extracted_pred,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct
                })
                
                if is_correct:
                    correct_predictions += 1
                total_samples += 1

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        eval_result = {"accuracy": accuracy}
        
        return records, eval_result 
