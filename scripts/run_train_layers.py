import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
import time
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf, ListConfig  # Import necessary for Hydra
import hydra # Import hydra

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
        ["nvidia-smi", "--query-gpu=memory.free ", "--format=csv,noheader,nounits"],
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


@hydra.main(config_path="../src/config", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    # Retrieve parameters from cfg
    dataset = cfg.data.name
    model_name = cfg.model_name
    num_shots = cfg.data.num_shot
    # use_extracted_cot_vector = cfg.use_extracted_cot_vector
    # extracted_cot_vector_path = getattr(cfg, "extracted_cot_vector_path", None)
    # static_shift_scale = getattr(cfg, "static_shift_scale", None)
    # static_mu_value = getattr(cfg, "static_mu_value", None)
    training_mode = getattr(cfg, "training-mode", "TRAIN_STUDENT_DIRECT_Q ")  # Default or read from cfg
    devices = getattr(cfg, "devices", None)
    use_cot_vector_type = getattr(cfg, "use_cot_vector_type", None)

    layer_train_mode = getattr(cfg, "layer_train_mode", "single_layer")
    dft_loss_signal = getattr(cfg, "dft_loss", False)
    if dft_loss_signal:
        cfg.ce_loss_weight = 1.0
        cfg.align_loss_weight = 0.1

    layers_to_iterate = []
    if layer_train_mode == "single_layer":
        start_layer = getattr(cfg, "start_layer", 0)
        end_layer = getattr(cfg, "end_layer", cfg.model_layers) 
        if end_layer is None:  # Fallback if model_layers is not set
            end_layer = 32
        layers_to_iterate = list(range(start_layer, end_layer))
        print(f"Evaluating in single_layer mode, from layer {start_layer} to {end_layer-1}")
    elif layer_train_mode == "combinations":
        shift_layers_combinations = getattr(cfg, "shift_layers_combinations", None)
        if not shift_layers_combinations:
            raise ValueError("shift_layers_combinations must be specified in train.yaml for 'combinations' mode.")
        layers_to_iterate = shift_layers_combinations
        print(f"Evaluating in combinations mode, with combinations: {shift_layers_combinations}")
    elif layer_train_mode == "dense":
        shift_layers_combinations = [list(range(0, 28))]
        layers_to_iterate = shift_layers_combinations
        print(f"Evaluating in combinations mode, with all layers injected.")
    else:
        raise ValueError(f"Unknown layer_train_mode: {layer_train_mode}. Must be 'single_layer', 'combinations' or 'dense'.")


    base_train_args_list = []
    if getattr(cfg, "train_args_additional", None):
        base_train_args_list.extend(shlex.split(cfg.train_args_additional))

    for current_layers in layers_to_iterate:
        # Determine how to format the layer string for runname and filename
        if isinstance(current_layers, ListConfig) or isinstance(current_layers, list):
            layers_str_for_name = "_".join(map(str, current_layers))
            only_shift_at_layer_arg = f"only_shift_at_layer='{str(current_layers)}'"  # Quoted JSON string
        elif isinstance(current_layers, int):
            layers_str_for_name = str(current_layers)
            only_shift_at_layer_arg = f"only_shift_at_layer={current_layers}"
        else:
            print("The type of shift layer in train.yaml is wrong.")

        # Construct the runname dynamically
        run_name_parts = [
            f"layer_train",
            f"{use_cot_vector_type}",
            # f"{dataset}",
            # f"{model_name}",
            f"{num_shots}shot",
            f"layers_{layers_str_for_name}",
            f"dft_{dft_loss_signal}",
        ]

        if cfg.data.use_self_cot:
            run_name_parts.append("sc")

        # if static_shift_scale is not None:
        #     run_name_parts.append(f"scale{static_shift_scale:.2f}".replace('.', '_'))
        # if static_mu_value is not None:
        #     run_name_parts.append(f"mu{static_mu_value:.2f}".replace('.', '_'))
        
        runname = "-".join(run_name_parts) # Use hyphen for runname for consistency with other runs

        # # Construct the record path for this specific run
        # extracted_name = os.path.basename(cfg.extracted_cot_vector_path).replace('.pt', '')
        # record_dir = os.path.join(src.paths.result_dir, "record", extracted_name)
        # os.makedirs(record_dir, exist_ok=True)
        
        # # The filename now includes the specific layer combination or single layer
        # if cfg.use_base_vector:
        #     record_path = os.path.join(record_dir,
        #                                f"use_base_vector_layers_{layers_str_for_name}_{cfg.static_mu_value}.json")
        # else:
        #     record_path = os.path.join(record_dir, f"{use_extracted_cot_vector_type}_layers_{layers_str_for_name}_{cfg.static_mu_value}.json")
        #
        # if os.path.exists(record_path):
        #     print(f"Skipping evaluation for layers {layers_str_for_name} as result already exists at {record_path}")
        #     continue

        print(f"Starting training {use_cot_vector_type} for layers {layers_str_for_name}...")

        # Build train_args for pipeline.py dynamically
        train_args_for_pipeline = base_train_args_list.copy()
        train_args_for_pipeline.append(f"data.name={dataset}")
        train_args_for_pipeline.append(f"data.num_shot={num_shots}")
        train_args_for_pipeline.append(f"training-mode={training_mode}")
        # train_args_for_pipeline.append(f"use_extracted_cot_vector={use_extracted_cot_vector}")
        # if extracted_cot_vector_path:
        #     train_args_for_pipeline.append(f"extracted_cot_vector_path={shlex.quote(extracted_cot_vector_path)}")
        # if static_shift_scale is not None:
        #     train_args_for_pipeline.append(f"static_shift_scale={static_shift_scale}")
        # if static_mu_value is not None:
        #     train_args_for_pipeline.append(f"static_mu_value={static_mu_value}")

        train_args_for_pipeline.append(only_shift_at_layer_arg)

        # Add encoder and peft configurations using Hydra's '+' syntax to append/override
        train_args_for_pipeline.append(f"+encoder={use_cot_vector_type}")
        train_args_for_pipeline.append(f"+peft={use_cot_vector_type}")

        # Construct the full command for pipeline.py
        command = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "pipeline.py"), # Corrected path to pipeline.py
            "-r", runname,
            "-d", dataset,
            "-m", model_name,
            "-s", str(num_shots),
            "-q", "3000",
            "--train",
            "--training-mode", training_mode,
            "--train-args", " ".join(train_args_for_pipeline),
            "--requires_memory", "22000",
            "--wait-devices-timeout", "100000"
        ]
        if devices:
            command.extend(["--devices", str(devices)])

        print(f"Running command: {' '.join(command)}")

        # Execute the command in the scripts directory, as pipeline.py expects to be run from project root.
        # Or, adjust the command to be run from the project root.
        # Let's adjust the command to be run from the project root for consistency.
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        process = subprocess.Popen(command, cwd=project_root, text=True)
        process.wait()

        if process.returncode != 0:
            print(f"Error running training for layer {current_layers}. See above for details.", file=sys.stderr)
            # Decide whether to exit or continue with other layers
            # For now, let's continue to allow other layers to run
        print(f"Finished training for layers {current_layers}.\n")

if __name__ == "__main__":
    main()
 