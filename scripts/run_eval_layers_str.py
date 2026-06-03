import sys
from pathlib import Path

# from src import shift_encoder
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 

import argparse
import os
import subprocess
import time
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf, ListConfig  # Import necessary for Hydra
import hydra # Import hydra
import src.paths

# debug
def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())

# debug
def get_avail_devices(devices, requires_memory=None):
    if not hasattr(get_avail_devices, "cached_requires_memory"):
        if requires_memory is None:
            raise ValueError("requires_memory must be provided on the first call.")
        get_avail_devices.cached_requires_memory = requires_memory

    effective_requires_memory = (
        requires_memory
        if requires_memory is not None
        else get_avail_devices.cached_requires_memory
    )

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    free_memory = result.stdout.strip().split("\n")
    free_gpus = [
        str(idx)
        for idx, mem in enumerate(free_memory)
        if int(mem) > effective_requires_memory
    ]

    if devices:
        return ",".join(set(devices.split(",")) & set(free_gpus))
    else:
        return ",".join(free_gpus)


@hydra.main(config_path="../src/config", config_name="eval_str.yaml", version_base=None)
def main(cfg: DictConfig):
    # Retrieve parameters from cfg
    dataset = cfg.data.name
    model_name = cfg.model_name
    num_shots = cfg.data.num_shot
    use_extracted_cot_vector = cfg.use_extracted_cot_vector
    extracted_cot_vector_path = getattr(cfg, "extracted_cot_vector_path", None)
    static_shift_scale = getattr(cfg, "static_shift_scale", None)
    static_mu_value = getattr(cfg, "static_mu_value", None)
    eval_mode = getattr(cfg, "eval_mode", "EVAL_WITH_COT_VECTOR_DIRECT_Q") # Default or read from cfg
    devices = getattr(cfg, "devices", None)
    use_extracted_cot_vector_type = getattr(cfg, "use_extracted_cot_vector_type", None)
    
    # Determine which layer evaluation mode to use
    layer_eval_mode = getattr(cfg, "layer_eval_mode", "single_layer")

    layers_to_iterate = []
    if layer_eval_mode == "single_layer":
        start_layer = getattr(cfg, "start_layer", 0)
        end_layer = getattr(cfg, "end_layer", cfg.model_layers) 
        if end_layer is None:  # Fallback if model_layers is not set
            end_layer = 28
        layers_to_iterate = list(range(start_layer, end_layer))
        print(f"Evaluating in single_layer mode, from layer {start_layer} to {end_layer-1}")
    elif layer_eval_mode == "combinations":
        shift_layers_combinations = getattr(cfg, "shift_layers_combinations", None)
        if not shift_layers_combinations:
            raise ValueError("shift_layers_combinations must be specified in eval.yaml for 'combinations' mode.")
        layers_to_iterate = shift_layers_combinations
        print(f"Evaluating in combinations mode, with combinations: {shift_layers_combinations}")
    elif layer_eval_mode == "dense":
        shift_layers_combinations = [list(range(0, 28))]
        layers_to_iterate = shift_layers_combinations
        print(f"Evaluating in combinations mode, with all layers injected.")
    else:
        raise ValueError(f"Unknown layer_eval_mode: {layer_eval_mode}. Must be 'single_layer', 'combinations' or 'dense'.")

    # Base eval arguments string from original eval.yaml, if any
    # This allows passing additional custom arguments from eval.yaml's 'eval_args_additional' field.
    base_eval_args_list = []
    if getattr(cfg, "eval_args_additional", None):
        base_eval_args_list.extend(shlex.split(cfg.eval_args_additional))

    for current_layers in layers_to_iterate:
        # Determine how to format the layer string for runname and filename
        if isinstance(current_layers, ListConfig) or isinstance(current_layers, list):
            layers_str_for_name = "_".join(map(str, current_layers))
            only_shift_at_layer_arg = f"only_shift_at_layer='{str(current_layers)}'"  # Quoted JSON string
        elif isinstance(current_layers, int):
            layers_str_for_name = str(current_layers)
            only_shift_at_layer_arg = f"only_shift_at_layer={current_layers}"
        else:
            print("The type of shift layer in eval.yaml is wrong.")

        # Construct the runname dynamically
        run_name_parts = [
            f"baseline-"
            f"layer_eval",
            f"{dataset}",
            f"{model_name}",
            f"{num_shots}shot",
            f"layers_{layers_str_for_name}"
        ]
        if use_extracted_cot_vector:
            run_name_parts.append("extractedCoT")
        if static_shift_scale is not None:
            run_name_parts.append(f"scale{static_shift_scale:.2f}".replace('.', '_'))
        if static_mu_value is not None:
            run_name_parts.append(f"mu{static_mu_value:.2f}".replace('.', '_'))
        
        runname = "-".join(run_name_parts) # Use hyphen for runname for consistency with other runs

        # Construct the record path for this specific run
        extracted_name = os.path.basename(cfg.extracted_cot_vector_path).replace('.pt', '')
        record_dir = os.path.join(src.paths.result_dir, "record", extracted_name)
        os.makedirs(record_dir, exist_ok=True)
        
        # The filename now includes the specific layer combination or single layer
        if cfg.use_base_vector:
            record_path = os.path.join(record_dir,
                                       f"use_base_vector_layers_{layers_str_for_name}_{cfg.static_mu_value}.json")
        else:
            record_path = os.path.join(record_dir, f"{cfg.use_extracted_cot_vector_type}_layers_{layers_str_for_name}_{cfg.static_mu_value}.json")

        if os.path.exists(record_path):
            print(f"Skipping evaluation for layers {layers_str_for_name} as result already exists at {record_path}")
            continue

        print(f"Starting evaluation for layers {layers_str_for_name}...")

        # Build eval_args for pipeline.py dynamically
        eval_args_for_pipeline = base_eval_args_list.copy()
        eval_args_for_pipeline.append(f"data.name={dataset}")
        eval_args_for_pipeline.append(f"data.num_shot={num_shots}")
        eval_args_for_pipeline.append(f"eval_mode={eval_mode}")
        eval_args_for_pipeline.append(f"use_extracted_cot_vector={use_extracted_cot_vector}")
        if extracted_cot_vector_path:
            eval_args_for_pipeline.append(f"extracted_cot_vector_path={shlex.quote(extracted_cot_vector_path)}")
        if static_shift_scale is not None:
            eval_args_for_pipeline.append(f"static_shift_scale={static_shift_scale}")
        if static_mu_value is not None:
            eval_args_for_pipeline.append(f"static_mu_value={static_mu_value}")
        
        # This is the key change: pass the current layer_idx or layer_combination to eval.py
        eval_args_for_pipeline.append(only_shift_at_layer_arg)

        # Add encoder and peft configurations using Hydra's '+' syntax to append/override
        eval_args_for_pipeline.append(f"+encoder={use_extracted_cot_vector_type}")
        eval_args_for_pipeline.append(f"+peft={use_extracted_cot_vector_type}")

        # Construct the full command for pipeline.py
        # command = [
        #     sys.executable,
        #     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "pipeline.py"), # Corrected path to pipeline.py
        #     "-r", runname,
        #     "-d", dataset,
        #     "-m", model_name,
        #     "-s", str(num_shots),
        #     "-q", "1000",
        #     "--eval",  # Ensure eval mode is enabled
        #     "--eval-mode", eval_mode,
        #     "--eval-args", " ".join(eval_args_for_pipeline)
        # ]
        # if devices:
        #     command.extend(["--devices", str(devices)])

        # print(f"Running command: {' '.join(command)}")


        # # Execute the command in the scripts directory, as pipeline.py expects to be run from project root.
        # # Or, adjust the command to be run from the project root.
        # # Let's adjust the command to be run from the project root for consistency.
        # project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        # process = subprocess.Popen(command, cwd=project_root, text=True)
        # process.wait()

        # if process.returncode != 0:
        #     print(f"Error running evaluation for layer {current_layers}. See above for details.", file=sys.stderr)
        #     # Decide whether to exit or continue with other layers
        #     # For now, let's continue to allow other layers to run
        # print(f"Finished evaluation for layers {current_layers}.\n")

        # 核心修改：直接构建 Hydra 兼容的参数列表
        hydra_overrides = [
            f"data.name={cfg.data.name}",
            f"data.num_shot={cfg.data.num_shot}",
            f"data.num_query_samples={cfg.data.num_query_samples}",
            f"eval_mode={cfg.eval_mode}",
            f"use_extracted_cot_vector={cfg.use_extracted_cot_vector}",
            f"extracted_cot_vector_path={cfg.extracted_cot_vector_path}",
            f"static_shift_scale={cfg.static_shift_scale}",
            f"static_mu_value={cfg.static_mu_value}",
            f"+encoder={cfg.use_extracted_cot_vector_type}", # 使用 + 前缀，确保 Hydra 能够正确合并配置
            f"+peft={cfg.use_extracted_cot_vector_type}", 
            f"only_shift_at_layer={current_layers}",  
            # f"run_name={cfg.run_name}"
        ]
        
        # 构建完整的命令行，直接调用 eval.py
        command = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "eval_str.py"),
        ] + hydra_overrides

        if cfg.devices:
            command.append(f"devices={cfg.devices}")

        print(f"Running command: {' '.join(command)}")

        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        process = subprocess.Popen(command, cwd=project_root, text=True)
        process.wait()

        if process.returncode != 0:
            print(f"Error running evaluation for layer {current_layers}. Exiting.", file=sys.stderr)
            sys.exit(1)

        print(f"Finished evaluation for layers {current_layers}.\n")

if __name__ == "__main__":
    main()
 


 