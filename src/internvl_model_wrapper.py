import os
import sys
import traceback
import types
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from testbed.models.model_base import ModelBase


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLModelWrapper(ModelBase):
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
        input_size = int(common_args.pop("input_size", 448))
        max_num_tiles = int(common_args.pop("max_num_tiles", 12))
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=support_models,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        config_input_size = getattr(self.model.config, "force_image_size", None)
        config_max_tiles = getattr(self.model.config, "max_dynamic_patch", None)
        self.input_size = int(config_input_size or input_size)
        self.max_num_tiles = int(config_max_tiles or max_num_tiles)

        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
        self.processor.padding_side = "left"

        # Text-only training/eval may not populate the InternVL image-context token id.
        # Use a sentinel so tensor comparison stays tensor-valued instead of collapsing
        # to a Python bool when the model forward checks `input_ids == img_context_token_id`.
        if getattr(self.model, "img_context_token_id", None) is None:
            token_id = None
            try:
                token_id = self.processor.convert_tokens_to_ids("<IMG_CONTEXT>")
            except Exception:
                token_id = None
            self.model.img_context_token_id = int(token_id) if token_id is not None else -1

        self._ensure_generation_support(self.model.language_model)
        self._patch_prepare_inputs_for_generation(self.model.language_model)

    @staticmethod
    def _ensure_generation_support(language_model) -> None:
        if isinstance(language_model, GenerationMixin):
            if getattr(language_model, "generation_config", None) is None:
                language_model.generation_config = GenerationConfig.from_model_config(
                    language_model.config
                )
            return

        current_cls = language_model.__class__
        patched_cls = type(
            f"{current_cls.__name__}WithGenerationMixin",
            (current_cls, GenerationMixin),
            {},
        )
        language_model.__class__ = patched_cls
        if getattr(language_model, "generation_config", None) is None:
            language_model.generation_config = GenerationConfig.from_model_config(
                language_model.config
            )

    @staticmethod
    def _patch_prepare_inputs_for_generation(language_model) -> None:
        original_prepare = language_model.prepare_inputs_for_generation

        def patched_prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
        ):
            if past_key_values is not None:
                try:
                    first_past = past_key_values[0][0]
                    if first_past is None:
                        past_key_values = None
                except Exception:
                    past_key_values = None

            return original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        language_model.prepare_inputs_for_generation = types.MethodType(
            patched_prepare_inputs_for_generation,
            language_model,
        )

    @staticmethod
    def _looks_like_formatted_prompt(text: str) -> bool:
        markers = (
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        )
        return any(marker in text for marker in markers)

    def _build_text_prompt(self, text: str) -> str:
        if self._looks_like_formatted_prompt(text):
            return text

        template = self.model.conv_template.copy()
        if hasattr(self.model, "system_message"):
            template.system_message = self.model.system_message
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    @staticmethod
    def _build_transform(input_size: int):
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> Tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: Optional[int] = None,
        image_size: Optional[int] = None,
        use_thumbnail: bool = False,
    ) -> List[Image.Image]:
        image_size = int(image_size or self.input_size)
        max_num = int(max_num or self.max_num_tiles)

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def _pil_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        transform = self._build_transform(self.input_size)
        processed_images = self._dynamic_preprocess(
            image,
            image_size=self.input_size,
            use_thumbnail=True,
            max_num=self.max_num_tiles,
        )
        pixel_values = [transform(tile) for tile in processed_images]
        return torch.stack(pixel_values)

    @torch.no_grad()
    def _generate_text_batch(
        self,
        prompts: List[str],
        **generate_args,
    ) -> List[str]:
        tokenizer = self.processor
        language_model = self.model.language_model
        device = next(language_model.parameters()).device

        formatted_prompts = [self._build_text_prompt(prompt) for prompt in prompts]
        model_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        sep_text = str(self.model.conv_template.sep).strip()
        eos_token_ids = [tokenizer.eos_token_id]
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_text)
        if sep_token_id is not None:
            eos_token_ids.append(sep_token_id)
        eos_token_ids = [token_id for token_id in eos_token_ids if token_id is not None]

        generated_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_ids,
            **generate_args,
        )

        input_length = input_ids.shape[1]
        responses = []
        for output_ids in generated_ids:
            generated_text_ids = output_ids[input_length:]
            text = tokenizer.decode(generated_text_ids, skip_special_tokens=True).strip()
            if sep_text and sep_text in text:
                text = text.split(sep_text, 1)[0].strip()
            responses.append(text)
        return responses

    @torch.no_grad()
    def _generate_image_batch(
        self,
        prompts: List[str],
        images: List[Any],
        **generate_args,
    ) -> List[str]:
        pixel_values_list = []
        num_patches_list = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type for InternVL: {type(image).__name__}")

            pixel_values = self._pil_to_pixel_values(image)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        device = next(self.model.language_model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = torch.cat(pixel_values_list, dim=0).to(device=device, dtype=model_dtype)
        return self.model.batch_chat(
            self.processor,
            pixel_values=pixel_values,
            questions=prompts,
            generation_config=dict(generate_args),
            num_patches_list=num_patches_list,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Any]] = None,
        **generate_args,
    ) -> List[str]:
        try:
            if images is None:
                images = [None] * len(prompts)

            if len(images) != len(prompts):
                raise ValueError(
                    f"Prompts/images length mismatch: {len(prompts)} prompts vs {len(images)} images."
                )

            has_image = [img is not None for img in images]
            if all(has_image):
                return self._generate_image_batch(prompts, images, **generate_args)
            if not any(has_image):
                return self._generate_text_batch(prompts, **generate_args)

            responses = [""] * len(prompts)
            for idx, (prompt, image) in enumerate(zip(prompts, images)):
                if image is None:
                    responses[idx] = self._generate_text_batch([prompt], **generate_args)[0]
                else:
                    responses[idx] = self._generate_image_batch([prompt], [image], **generate_args)[0]
            return responses
        except Exception as exc:
            print(f"[InternVL generate error] {exc}")
            traceback.print_exc()
            return [""] * len(prompts)

    @property
    def device(self):
        return next(self.model.language_model.parameters()).device
