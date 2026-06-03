import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Optional, Union, Tuple

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from testbed.models.model_base import ModelBase

class QwenModelWrapper(ModelBase):
    def __init__(
        self,
        model_root: str,
        processor_class: type,
        model_class: type,
        support_models: Optional[List[str]] = None,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        # Call super().__init__ first to properly initialize the Module and load model/processor
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=support_models,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        # ModelBase has already set self.model, self.processor, and self.config
        # We can re-assign model_name for clarity if needed, but it should be set by ModelBase too.

    def _build_tokenizer_kwargs(self) -> Dict[str, Any]:
        tokenizer_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if "internlm" in getattr(self, "model_name", "").lower():
            # InternLM chat prompts already contain BOS/chat special tokens after
            # apply_chat_template(tokenize=False); adding them again can break generation.
            tokenizer_kwargs["add_special_tokens"] = False
        return tokenizer_kwargs

    def _generate_single_prompt(
        self,
        prompt: str,
        **generate_args,
    ) -> str:
        model_inputs = self.processor([prompt], **self._build_tokenizer_kwargs()).to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **generate_args,
            pad_token_id=self.processor.eos_token_id,
        )

        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0]
        start_index = input_length if output_ids.shape[0] > input_length else 0
        generated_text_ids = output_ids[start_index:]
        return self.processor.decode(generated_text_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        processor_args: Optional[Dict[str, Any]] = None,
        return_inputs: bool = False,
        return_generated_ids: bool = False,
        **generate_args,
    ) -> Optional[List[str]]:
        if "internlm" in getattr(self, "model_name", "").lower() and len(prompts) > 1:
            return [
                self._generate_single_prompt(prompt, **generate_args)
                for prompt in prompts
            ]

        # Use self.processor and self.model which are instantiated by ModelBase
        model_inputs = self.processor(prompts, **self._build_tokenizer_kwargs()).to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **generate_args,
            pad_token_id=self.processor.eos_token_id,
        )

        input_length = model_inputs.input_ids.shape[1]
        decoded_responses = []
        for i, output_ids in enumerate(generated_ids):
            start_index = input_length if output_ids.shape[0] > input_length else 0
            generated_text_ids = output_ids[start_index:]
            decoded_responses.append(self.processor.decode(generated_text_ids, skip_special_tokens=True))
        
        return decoded_responses

    def process_input(
        self,
        images: Any,
        text: Union[List[str], List[List[str]]],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        # Use self.processor which is instantiated by ModelBase
        return text

    @property
    def default_prompt_template(self) -> str:
        return ""

    @property
    def device(self):
        return self.model.device 
    
