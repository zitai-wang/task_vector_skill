import re
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import evaluate
import os # Import os module
from tqdm import tqdm # Import tqdm
import random # Import random
import enum
import torch
import json

from src.dataset_utils.interface import DatasetBase
# import importlib
# interface = importlib.import_module('interface')
# from .interface import DatasetBase
from testbed.data import prepare_dataloader, prepare_input
from testbed.models.model_base import ModelBase
from difflib import SequenceMatcher

def find_lcs_token_ids(teacher_tokens, student_tokens):
    """
    找到 teacher 和 student answer token 序列的最长公共子序列（LCS）
    返回 LCS 的 token ID 列表
    """
    matcher = SequenceMatcher(None, teacher_tokens, student_tokens, autojunk=False)
    lcs_tokens = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            lcs_tokens.extend(teacher_tokens[i1:i2])
    return lcs_tokens

def build_new_mask_from_lcs(original_ids, lcs_tokens):
    """
    在原始 token 序列中找到 LCS 的位置，返回新的 mask
    """
    mask = [False] * len(original_ids)
    lcs_len = len(lcs_tokens)
    if lcs_len == 0:
        return mask

    # 滑动窗口匹配 LCS
    for i in range(len(original_ids) - lcs_len + 1):
        if original_ids[i:i+lcs_len] == lcs_tokens:
            for j in range(i, i + lcs_len):
                mask[j] = True
            break  # 只匹配第一次出现的位置
    return mask

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

# only for MMLU-Pro
class MATHDataset(DatasetBase):
    support_datasets = ["math_dataset"]

    def __init__(self, data_cfg: DictConfig, model_processor: Any = None, model_name: Optional[str] = None) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        # local_path = "/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/"
        local_path = "/data1/wzy/MATH/"
        train_path = "math_train_5000_subset"
        test_path = "math_test_1000_subset"
        easy_train_path = "math_train_subset_easy"
        easy_test_path = "math_test_subset_easy"
        hard_train_path = "math_train_subset_hard"
        hard_test_path = "math_test_subset_hard"
        
        # Check if 'subset' key exists in data_cfg, default to None or "normal" if not present
        subset = data_cfg.get("subset", "normal")
        
        if subset == "easy":
            test_path = easy_test_path
            train_path = easy_train_path
            self.subset_type = "easy"
        elif data_cfg.subset == "hard":
            test_path = hard_test_path
            train_path = hard_train_path
            self.subset_type = "hard"
        else:
            self.subset_type = "all"
        self._query_set = Dataset.load_from_disk(local_path + test_path)
        if len(self._query_set) > 1000:
            self._query_set = self._query_set.select(range(1000))
        self._support_set = Dataset.load_from_disk(local_path + train_path)
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
                f"math Dataset initialized: Training set limited to {len(self._support_set)} samples by num_query_samples.")

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
                self._self_cot_data[question] = {
                    'self_cot': data['self_cot'],
                    'is_correct': data['is_correct'],
                    'gt_answer': data['gt_answer']
                }
        
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
                    sample_with_self_cot['gt_answer'] = self_cot_info['gt_answer']
                    filtered_samples.append(sample_with_self_cot)
        
        self._filtered_support_set = filtered_samples
        print(f"Loaded {len(self._self_cot_data)} Self-CoT samples, filtered to {len(self._filtered_support_set)} correct samples")
        
        # Update support set to use filtered data
        if self._filtered_support_set:
            self._support_set = Dataset.from_list(self._filtered_support_set)

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

    def extract_answer(self, prediction: str) -> str:
        # mmlu-pro answers are typically in the format "The answer is (I)."
        # We need to extract the numerical answer.
        patterns = [
            r"\\boxed{(.+)}",
            r'answer is \(?([A-Z])\)?',
            r"\boxed{(.+)}",
            r"answer is \(?([A-Z])\)?",
            r'.*[aA]nswer:\s*([A-Z])',
        ]

        # 尝试匹配每个模式
        for pattern in patterns:
            match = re.search(pattern, prediction)
            if match:
                return match.group(1).strip()
            else:
                final_pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
                match = re.search(final_pattern, prediction, re.DOTALL)
                if match:
                    return match.group(0)
                else:
                    return None


    def _extract_final_answer_from_full_cot(self, full_cot_answer: str) -> str:
        # Extracts just the numerical answer (e.g., "123") from "#### 123" or "\boxed{123}"
        return self.extract_answer(full_cot_answer)

    def _extract_cot_only(self, full_cot_answer: str) -> str:
        # First, extract the part before "####" if it exists. This is the main CoT part.
        cot_part_before_hash = full_cot_answer
        match_hash = re.search(r"\boxed{(.+)}", full_cot_answer)
        if match_hash:
            cot_part_before_hash = full_cot_answer[:match_hash.start()].strip()
            cot_part_after_hash = full_cot_answer[match_hash.start():].strip()

        # Then, extract the final numerical answer from the full string.
        # This reuses the logic that extracts the boxed or #### answer.
        numerical_final_answer = self.extract_answer(full_cot_answer)

        if numerical_final_answer:
            # Find the last occurrence of the numerical final answer within the CoT part.
            # This allows us to truncate right before the answer itself.
            last_answer_idx = cot_part_after_hash.rfind(numerical_final_answer)
            if last_answer_idx != -1:
                # Truncate the CoT part right before the numerical answer
                return cot_part_before_hash.strip() + cot_part_after_hash[:last_answer_idx].strip()
                
        # Fallback: if numerical_final_answer is not found or no specific sentence contains it,
        # return the CoT part before #### (which was already handled by cot_part_before_hash).
        return cot_part_before_hash.strip()

    def _format_single_example(self, example: Dict, use_cot: bool, model_name: str, for_teacher: bool = False) -> str:
        question = example["problem"]
        full_cot_answer = example["solution"] # This contains the full CoT and "#### final_answer"

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


    def _create_qwen_chat_template(self, question: str, use_cot: bool,
                                   few_shot_examples: List[Dict] = None, dataset_state: DatasetState = DatasetState.EVAL_BASELINE) -> List[Dict]:
        messages = []

        system_prompt = ""
        # Determine system prompt based on whether final query expects CoT
        if dataset_state in [DatasetState.EVAL_BASELINE, DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT, DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q]:
            if use_cot:
                system_prompt = "You are a helpful and precise assistant for solving math problems. Please reason step by step, and put your final answer within \\boxed{}."
            else:
                system_prompt = "You are a helpful and precise assistant for solving math problems. Put your answer within \\boxed{}."
        elif dataset_state in [DatasetState.TRAIN_TEACHER, DatasetState.TRAIN_TEACHER_SELF_COT]:
            # Teacher model: expects full CoT and final answer from the dataset.
            # The system prompt should indicate a step-by-step reasoning with a boxed answer.
            system_prompt = "You are a helpful and precise assistant for solving math problems. Please reason step by step, and put your final answer within \\boxed{}."
        elif dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_DIRECT_Q, 
                              DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
            # Student model: for training, we now ask for direct numerical answer, no \boxed{} for simplicity.
            # The CoT effect comes from the shift vector.
            system_prompt = "You are a helpful and precise assistant for solving math problems. Output the final numerical answer."
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add few-shot examples for student model (TRAIN_STUDENT_ONESHOT) or evaluation (EVAL_BASELINE etc.)
        if few_shot_examples and (dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.EVAL_BASELINE, 
                                                    DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT,
                                                    DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT]):
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["problem"]})
                # For student few-shot training, we want the final numerical answer ONLY, no \boxed{}.
                # For eval, if EVAL_BASELINE or EVAL_WITH_COT_VECTOR_ONESHOT, we still use \boxed{} as that's the expected output format for final answer.
                formatted_example_answer = example["solution"]
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
                example_cot_sequence = example['solution']
                if dataset_state == DatasetState.TRAIN_TEACHER_SELF_COT and 'self_cot' in example:
                    # Use Self-CoT + final answer for TRAIN_TEACHER_SELF_COT
                    self_cot_with_answer = f"{example['self_cot']}\\boxed{{example['gt_answer']}})."
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

    from transformers import PreTrainedTokenizerBase

    def get_answer_token_mask(self,
            offset_mapping: list,
            full_prompt: str,
            final_answer: str,
            tokenized_input_ids: list[int],
    ) -> list[bool]:
        """
        返回一个布尔列表，标记哪些 token 属于 final_answer。
        """
        if not final_answer:
            return [False] * len(tokenized_input_ids)

        # 找到 final_answer 在 full_prompt 中的字符级起止位置
        start_char = full_prompt.rfind(final_answer)
        if start_char == -1:
            return [False] * len(tokenized_input_ids)

        end_char = start_char + len(final_answer)

        # 根据 offset_mapping 找到对应的 token 索引
        mask = [False] * len(tokenized_input_ids)
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start >= end_char:
                break
            if token_end > start_char:
                mask[idx] = True

        return mask

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        # This collate function will be used for training dataloaders.
        # It needs to produce: prefix_inputs, student_inputs, student_labels, images
        # images will be empty lists for Qwen.

        teacher_prompts = []
        student_input_ids_list = [] # Will store tensor of full input_ids for student
        student_prompt_lengths = [] # Will store len(prompt_ids) directly for masking
        images = [[] for _ in batch_data] # Empty images for Qwen
        teacher_final_answer_token_ids_list = [] # New list to store tokenized final answers for each sample
        teacher_final_answer_list = []
        cot_texts = []

        for item in batch_data:
            question = item["problem"]
            full_cot_answer = item["solution"]
            cot_texts.append(full_cot_answer)
            final_answer = self._extract_final_answer_from_full_cot(full_cot_answer)
            teacher_final_answer_list.append(final_answer)

            # Determine teacher dataset state based on current dataset state
            teacher_dataset_state = self.dataset_state
            if self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                teacher_dataset_state = DatasetState.TRAIN_TEACHER_SELF_COT
            elif self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_DIRECT_Q]:
                teacher_dataset_state = DatasetState.TRAIN_TEACHER

            # Teacher input: Q + CoT + Answer (GT-CoT or Self-CoT)
            teacher_prompt_messages = self._create_qwen_chat_template(
                question=question,
                use_cot=True,
                few_shot_examples=[item],  # Pass the full item to access self_cot if available
                dataset_state=teacher_dataset_state
            )
            
            teacher_prompt_text = self.model_processor.apply_chat_template(teacher_prompt_messages, tokenize=False, add_generation_prompt=False)
            teacher_prompts.append(teacher_prompt_text)

            # Tokenize the final answer separately (without special tokens)
            final_answer_ids = self.model_processor(str(final_answer), add_special_tokens=False).input_ids
            teacher_final_answer_token_ids_list.append(final_answer_ids) # Store for later use
            
            # Student prompt: depends on training mode
            student_prompt_messages = None
            if self.dataset_state in [DatasetState.TRAIN_STUDENT_ONESHOT, DatasetState.TRAIN_STUDENT_ONESHOT_SELF_COT]:
                if len(self._support_set) < 1:
                    raise ValueError("Not enough examples in training set for 1-shot learning.")
                one_shot_example = random.sample(list(self._support_set), 1)[0]
                
                student_prompt_messages = self._create_qwen_chat_template(
                    question=question,
                    use_cot=False, # Student is trained to output direct answer
                    few_shot_examples=[one_shot_example],
                    dataset_state=self.dataset_state
                )
            elif self.dataset_state in [DatasetState.TRAIN_STUDENT_DIRECT_Q, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                student_prompt_messages = self._create_qwen_chat_template(
                    question=question,
                    use_cot=False, # Student is trained to output direct answer
                    few_shot_examples=None,
                    dataset_state=self.dataset_state
                )
            else:
                raise ValueError(f"Invalid dataset state for training collate_fn: {self.dataset_state}")

            student_prompt_text = self.model_processor.apply_chat_template(
                student_prompt_messages,
                tokenize=False,
                add_generation_prompt=True # Ensures the assistant token is added
            )

            # For student training, we now aim for direct numerical answer output.
            
            # Tokenize prompt and answer separately
            prompt_ids = self.model_processor(student_prompt_text, add_special_tokens=True).input_ids
            student_prompt_lengths.append(len(prompt_ids)) # Store prompt length here

            answer_ids = self.model_processor(str(final_answer), add_special_tokens=False).input_ids # Tokenize final_answer directly

            # Construct the full sequence of IDs, including EOS token
            full_ids_for_sample = prompt_ids + answer_ids + [self.model_processor.eos_token_id]
            student_input_ids_list.append(torch.tensor(full_ids_for_sample, dtype=torch.long))

        # Tokenize all teacher prompts in batch
        teacher_inputs = self.model_processor(
            teacher_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
            return_offsets_mapping=True,
            add_special_tokens=True # Already handled by chat template, but ensure consistency
        )

        # for LORA
        batch_teacher_cot_mask = torch.zeros_like(
            teacher_inputs["input_ids"], dtype=torch.bool
        )  # [B, L_teacher]

        for i, cot_str in enumerate(cot_texts):
            if not cot_str:  # 防止空串
                continue

            # 整句 token 序列
            full_ids = teacher_inputs["input_ids"][i].tolist()
            # CoT 子序列
            cot_ids = self.model_processor(
                cot_str, add_special_tokens=False
            ).input_ids

            if len(cot_ids) == 0:
                continue

            # 在 full_ids 里找 cot_ids 的「最早一次出现」
            L = len(full_ids)
            K = len(cot_ids)
            start = -1
            for pos in range(L - K + 1):
                if full_ids[pos: pos + K] == cot_ids:
                    start = pos
                    break
            if start == -1:  # 没找到就跳过
                continue

            # end = start + K
            batch_teacher_cot_mask[i, start:] = True

        # Create teacher_answer_mask here, after teacher_inputs are padded/tokenized
        batch_teacher_answer_mask = torch.zeros_like(teacher_inputs["input_ids"], dtype=torch.bool)

        for i, (off_mapping, full_prompt, final_answer) in enumerate(zip(teacher_inputs.encodings, teacher_prompts, teacher_final_answer_list)):
            mask = self.get_answer_token_mask(
                off_mapping.offsets,
                full_prompt,
                final_answer,
                teacher_inputs["input_ids"][i].tolist()
            )
            batch_teacher_answer_mask[i] = torch.tensor(mask, dtype=torch.bool)

        # Pad student input sequences to max_seq_len
        student_inputs_padded = torch.nn.utils.rnn.pad_sequence(
            student_input_ids_list, batch_first=True, padding_value=self.model_processor.pad_token_id,
            padding_side="left",
        )
        
        # Truncate if necessary (should be handled by max_length in tokenizer, but defensive)
        if student_inputs_padded.shape[1] > max_seq_len:
            student_inputs_padded = student_inputs_padded[:, :max_seq_len]

        # Create attention mask - ensure EOS token is included even if pad_token_id == eos_token_id
        student_attention_mask = torch.zeros_like(student_inputs_padded, dtype=torch.long)
        for i, original_ids_for_sample in enumerate(student_input_ids_list):
            num_leading_pads = student_inputs_padded.shape[1] - len(original_ids_for_sample)
            student_attention_mask[i, num_leading_pads:] = 1

        student_inputs = {
            "input_ids": student_inputs_padded,
            "attention_mask": student_attention_mask
        }

        # Create student_labels: Initialize all to -100
        student_labels = torch.full_like(student_inputs["input_ids"], -100, dtype=torch.long)

        for i, prompt_len_val in enumerate(student_prompt_lengths):
            # The non-padded full sequence for this sample
            full_ids_for_sample_original = student_input_ids_list[i]

            # Determine how many padding tokens are at the beginning of the padded sequence
            # (This is due to left-padding)
            current_actual_seq_len = student_inputs["attention_mask"][i].sum().item()
            num_padding_tokens = student_inputs_padded.shape[1] - current_actual_seq_len

            # Calculate the start index of the answer in the padded sequence
            # The answer starts after the prompt in the original sequence
            answer_start_original_idx = prompt_len_val
            answer_start_padded_idx = num_padding_tokens + answer_start_original_idx

            # Copy the answer tokens to the student_labels
            # Make sure we don't go out of bounds of the padded sequence
            # This copies everything from `answer_start_padded_idx` to the end of the actual (non-padded) sequence
            student_labels[i, answer_start_padded_idx:-1] = \
                student_inputs_padded[i, answer_start_padded_idx:-1]

        student_labels = (student_labels != -100) & student_inputs["attention_mask"].bool()
        for i in range(len(batch_data)):
            masked_teacher_input = teacher_inputs["input_ids"][i][batch_teacher_answer_mask[i].bool()].tolist()
            masked_student_input = student_inputs["input_ids"][i][student_labels[i].bool()].tolist()

            if len(masked_teacher_input) != len(masked_student_input):
                # 找到最长公共子序列
                lcs_tokens = find_lcs_token_ids(masked_teacher_input, masked_student_input)

                # 用 LCS 构造新的 answer mask
                new_teacher_mask = torch.tensor(
                    build_new_mask_from_lcs(teacher_inputs["input_ids"][i].tolist(), lcs_tokens),
                    dtype=torch.bool
                )

                new_student_mask = torch.tensor(
                    build_new_mask_from_lcs(student_inputs["input_ids"][i].tolist(), lcs_tokens),
                    dtype=torch.bool
                )
                batch_teacher_answer_mask[i] = new_teacher_mask
                student_labels[i] = new_student_mask
        return {
            "prefix_inputs": teacher_inputs,
            "teacher_answer_mask": batch_teacher_answer_mask,
            "teacher_cot_mask": batch_teacher_cot_mask, # New output
            "student_inputs": student_inputs,
            "student_labels": student_labels,
            "images": images,
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
            question = item["problem"]
            ground_truth = item["solution"]

            messages = self._create_qwen_chat_template(question=question, use_cot=True,
                                                       few_shot_examples=sampled_few_shot_examples,
                                                       dataset_state=self.dataset_state)
            batch_messages.append(messages)
            ground_truths.append(self.extract_answer(ground_truth))

        return{
            "batch_messages": batch_messages,
            "ground_truths": ground_truths,
        }

    def validation_dataloader(self, batch_size: int) -> Any:
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

    def collate_fn_for_eval(self, batch_data: List[Dict]):
        # 定义选项字母（A, B, C, ...）
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

        processed_batch = []
        for item in batch_data:
            # 复制原始项
            new_item = item.copy()

            # 将选项列表转换为格式化字符串
            options = item['options']
            formatted_options = []
            for i, option in enumerate(options):
                if i < len(option_letters):
                    formatted_options.append(f"{option_letters[i]}. {option}")
                else:
                    # 如果选项超过字母数量，使用数字
                    formatted_options.append(f"{i + 1}. {option}")

            # 用逗号连接所有选项
            new_item['options'] = "; ".join(formatted_options)

            processed_batch.append(new_item)

        # 将处理后的批次转换为字典形式
        keys = processed_batch[0].keys()
        collated = {}
        for key in keys:
            collated[key] = [d[key] for d in processed_batch]

        return collated

    def eval(
        self,
        eval_cfg: DictConfig,
        model: ModelBase,
    ) -> Tuple[List[Dict], Dict]:
        generation_args = eval_cfg.generation_args
        batch_size = eval_cfg.batch_size
        use_cot = eval_cfg.use_cot # New config to switch between direct and CoT
        num_shot = eval_cfg.data.num_shot  # Get num_shot from config

        # Set dataset state for evaluation (baseline or with CoT vector)
        if getattr(eval_cfg, "eval_mode", None) == "eval_with_cot_vector_oneshot":
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT
        elif getattr(eval_cfg, "eval_mode", None) == "eval_with_cot_vector_direct_q":
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q
        else:
            self.dataset_state = DatasetState.EVAL_BASELINE

        # for baseline, use cot means direct q, not use cot means n shot
        if self.dataset_state == DatasetState.EVAL_BASELINE and use_cot:
            num_shot = 0

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
        
        # Apply max_samples limit if specified
        if getattr(eval_cfg, "max_samples", 1000) > 0 and eval_cfg.max_samples < len(dataset_to_load):
            dataset_to_load = dataset_to_load.select(range(eval_cfg.max_samples))
            print(f"Limiting to {len(dataset_to_load)} samples based on max_samples configuration.")


        dataloader = DataLoader(
            dataset_to_load,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset_to_load.collate_fn if hasattr(dataset_to_load, 'collate_fn') else None
            # collate_fn=lambda batch: self.collate_fn_for_eval(batch),
        )

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating MATH DATASET({self.subset_type})")):
            questions = batch["problem"]
            ground_truth_answers = batch["solution"]

            inputs = []
            for idx, question in enumerate(questions):
                # For eval, the few-shot examples are passed to the chat template.
                # The `use_cot` for template creation depends on `eval_cfg.use_cot` for baseline
                # or the `eval_mode` for cot vector eval.

                messages = self._create_qwen_chat_template(question=question, use_cot=True,
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
                    # 这里新增
                    # predictions = None
                else:
                    raise exception

            if predictions is None:
                # Handle OOM or other failures
                print(f"Skipping batch {batch_idx} due to prediction failure (e.g., OOM). ")
                continue

            for i, pred in enumerate(predictions):
                question = questions[i]
                ground_truth = ground_truth_answers[i]

                extracted_pred = self.extract_answer(pred)
                extracted_ground_truth = self.extract_answer(ground_truth)

                is_correct = (extracted_pred == extracted_ground_truth)
                
                records.append({
                    "question": question,
                    "model_output": pred,
                    "extracted_prediction": extracted_pred,
                    "ground_truth": ground_truth,
                    "extracted_ground_truth": extracted_ground_truth,
                    "is_correct": is_correct
                })
                
                if is_correct:
                    correct_predictions += 1
                total_samples += 1

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        eval_result = {"accuracy": accuracy}
        
        return records, eval_result 
