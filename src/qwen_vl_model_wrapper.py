import sys
import torch
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Any, Dict, List, Optional, Union, Tuple
from testbed.models.model_base import ModelBase


class QwenVLModelWrapper(ModelBase):
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
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=support_models,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        # 检查模型结构，用于后续 Hook 注入
        if hasattr(self.model, "visual") and hasattr(self.model, "model"):
            self.internal_llm = self.model.model 
        else:
            print("[QwenVLWrapper] Warning: Standard Qwen-VL structure not found.")

    @staticmethod
    def _looks_like_formatted_prompt(text: str) -> bool:
        markers = (
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
        return any(marker in text for marker in markers)

    def _prepare_prompt(self, text: str, image: Optional[Any]) -> str:
        if self._looks_like_formatted_prompt(text):
            return text

        user_content = []
        if image is not None:
            if isinstance(image, (list, tuple)):
                for single_image in image:
                    user_content.append({"type": "image", "image": single_image})
            else:
                user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_batch(
        self,
        prompts: List[str],
        images: Optional[List[Any]],
        **generate_args,
    ) -> List[str]:
        processor_inputs = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if images is not None and any(img is not None for img in images):
            processor_inputs["images"] = images

        inputs = self.processor(**processor_inputs)
        final_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        n_beams = generate_args.pop("num_beams", 1)
        d_sample = generate_args.pop("do_sample", False)
        generate_args.pop("processor_args", None)
        generate_args.pop("model_args", None)

        generated_ids = self.model.generate(
            **final_inputs,
            num_beams=n_beams,
            do_sample=d_sample,
            **generate_args,
        )

        input_ids = final_inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Any]] = None,
        **generate_args,
    ) -> List[str]:
        if images is None:
            images = [None] * len(prompts)

        formatted_prompts = [
            self._prepare_prompt(text, images[i]) for i, text in enumerate(prompts)
        ]

        try:
            # Mixed image/no-image batches are easier to handle sample by sample.
            if any(img is None for img in images) and any(img is not None for img in images):
                outputs = []
                for prompt, image in zip(formatted_prompts, images):
                    batch_images = None if image is None else [image]
                    outputs.extend(
                        self._generate_batch([prompt], batch_images, **dict(generate_args))
                    )
                return outputs

            batch_images = None if all(img is None for img in images) else images
            return self._generate_batch(formatted_prompts, batch_images, **generate_args)
        except Exception as e:
            print(f" [GENERATE ERROR]: {e}")
            traceback.print_exc()
            return [""] * len(prompts)

    @property
    def device(self):
        return self.model.device    
