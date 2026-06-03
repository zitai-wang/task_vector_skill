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


@hydra.main(config_path="../src/config", config_name="eval.yaml", version_base=None)
def main(cfg: DictConfig):
    # Retrieve parameters from cfg
    dataset = cfg.data.name
    model_name = cfg.model_name
    num_shots = cfg.data.num_shot
    use_extracted_cot_vector = False
    extracted_cot_vector_path = None
    static_shift_scale = None
    static_mu_value = None
    eval_mode = getattr(cfg, "eval_mode", "EVAL_WITH_COT_VECTOR_DIRECT_Q") # Default or read from cfg
    devices = getattr(cfg, "devices", None)
    use_extracted_cot_vector_type = None


    # Base eval arguments string from original eval.yaml, if any
    # This allows passing additional custom arguments from eval.yaml's 'eval_args_additional' field.
    base_eval_args_list = []
    if getattr(cfg, "eval_args_additional", None):
        base_eval_args_list.extend(shlex.split(cfg.eval_args_additional))


    # Construct the runname dynamically
    run_name_parts = [
        f"baseline-"
        f"{dataset}",
        f"{model_name}",
        f"{num_shots}shot",
    ]

    runname = "-".join(run_name_parts) # Use hyphen for runname for consistency with other runs

    record_dir = os.path.join(src.paths.result_dir, "record", runname)
    os.makedirs(record_dir, exist_ok=True)

    record_path = os.path.join(record_dir, f"debug.json")

    if os.path.exists(record_path):
        print(f"Skipping evaluation as result already exists at {record_path}")
        return

    print(f"Starting evaluation for DEBUG...")

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
    # eval_args_for_pipeline.append(only_shift_at_layer_arg)

    # Add encoder and peft configurations using Hydra's '+' syntax to append/override
    eval_args_for_pipeline.append(f"+encoder=licv")
    eval_args_for_pipeline.append(f"+peft=licv")

    # Construct the full command for pipeline.py
    command = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "pipeline.py"), # Corrected path to pipeline.py
        "-r", runname,
        "-d", dataset,
        "-m", model_name,
        "-s", str(num_shots),
        "-q", "1000",
        "--eval",  # Ensure eval mode is enabled
        "--eval-mode", eval_mode,
        "--eval-args", " ".join(eval_args_for_pipeline)
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
        print(f"Error running evaluation for DEBUG. See above for details.", file=sys.stderr)
        # Decide whether to exit or continue with other layers
        # For now, let's continue to allow other layers to run
    print(f"Finished evaluation for DEBUG.\n")

if __name__ == "__main__":
    main()
 