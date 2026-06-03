# -*- coding: utf-8 -*-

import json
import torch
import os
import re
import evaluate
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paths
from utils import *
from dataset_utils import dataset_mapping, DatasetBase
from qwen_model_wrapper import QwenModelWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# import sys
# print("--- Debugging Environment Info ---")
# print(f"Python Version: {sys.version}")
# print(f"Python Executable: {sys.executable}")
# print("----------------------------------")
# 
import re
import sys

def check_invisible_chars(content):
    # 
    invisible = re.findall(r'[^\x20-\x7E\t\n\r]', content)
    if invisible:
        print(f" {len(invisible)} :")
        for i, char in enumerate(set(invisible)):
            print(f" {i+1}: Unicode U+{ord(char):04X}")
        return True
    return False

# 
with open(__file__, 'r', encoding='utf-8') as f:
    content = f.read()

if check_invisible_chars(content):
    print("")
    # 
    cleaned_content = re.sub(r'[^\x20-\x7E\t\n\r]', '', content)
    with open(__file__ + '.cleaned.py', 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    print(":", __file__ + '.cleaned.py')
    sys.exit(1)

@hydra.main(config_path="config", config_name="eval.yaml", version_base=None)
def main(cfg: DictConfig):
    # initialize dataset and variables
    runname = get_expand_runname(cfg)
    is_icl = cfg.ckpt_path is None
    cfg.data.is_icl = is_icl

    dataset: DatasetBase = dataset_mapping[cfg.data.name](cfg.data)
    record_dir = os.path.join(paths.result_dir, "record", runname)
    if is_icl:
        record_path = os.path.join(record_dir, f"{cfg.data.num_shot}shot.json")
    else:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {cfg.ckpt_path} not found.")

        epoch = re.findall(r"\d+", os.path.basename(cfg.ckpt_path))
        if len(epoch) != 1:
            raise ValueError(
                f"Invalid checkpoint path {cfg.ckpt_path}. It should contain a single number in basename for epoch."
            )
        epoch = int(epoch[0])
        record_path = os.path.join(record_dir, f"epoch-{epoch}.json")

    os.makedirs(record_dir, exist_ok=True)

    if cfg.resume:
        if os.path.exists(record_path):
            print(f"Found exist record {record_path}, skip...")
            return

    # load model and inference
    device = torch.device("cuda")
    
    if "qwen2.5-math-7b-instruct" in cfg.model_name:
        qwen_raw_model, qwen_tokenizer = build_model(cfg)
        model = QwenModelWrapper(qwen_raw_model, qwen_tokenizer).to(device, eval(cfg.dtype))
        is_icl = True
    else:
        model = build_model(cfg).to(device, eval(cfg.dtype))

    if not is_icl:
        encoder = hydra.utils.instantiate(cfg.encoder.cls, lmm=model).to(
            device, eval(cfg.dtype)
        )

        print(f"Loading from pretrained: {cfg.ckpt_path}")
        load_from_pretrained(cfg.ckpt_path, model, encoder)
        encoder.eval()
        hooks = encoder.register_shift_hooks()

    model.eval()

    Path(record_path).touch()
    try:
        result, eval_result = dataset.eval(cfg, model)
        print(f"Evaluation result for {runname}: {eval_result}")
        config = {"eval_args": OmegaConf.to_container(cfg, resolve=True)}
        if os.path.exists(os.path.join(record_dir, "config.json")):
            with open(os.path.join(record_dir, "config.json")) as f:
                config["train_args"] = json.load(f)

        evaluate.save(
            record_path,
            **config,
            eval_result=eval_result,
            records=result,
        )

    finally:
        if os.path.exists(record_path) and os.path.getsize(record_path) == 0:
            os.remove(record_path)


if __name__ == "__main__":
    main()
