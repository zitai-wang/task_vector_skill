import argparse
import os
import re
import subprocess
import sys
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shlex
import paths

# debug
def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())


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

    # Check if a specific 'cpu' device is requested via the 'devices' argument
    # If 'cpu' is explicitly in devices, we assume CPU mode
    if devices and 'cpu' in devices.lower():
        return "cpu" # Special string to indicate CPU usage

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
        # Ensure we only return intersection with actual available GPUs if not in CPU mode
        return ",".join(set(devices.split(",")) & set(free_gpus))
    else:
        return ",".join(free_gpus)


def run_train(
    runname,
    dataset,
    num_query_sample,
    num_shot,
    model_name,
    train_args,
    devices,
    training_mode,
):
    try:
        # Determine CUDA_VISIBLE_DEVICES based on available devices
        visible_devices = get_avail_devices(devices)
        if visible_devices == "cpu":
            os_environ_cuda = "" # Empty string means CPU for PyTorch
        else:
            os_environ_cuda = visible_devices

        process = subprocess.Popen(
            [
                sys.executable,
                "train.py",
            ]
            + merge_args(
                [
                    f"runname={runname}",
                    f"model_name={model_name}",
                    f"data.num_query_samples={num_query_sample}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.training_mode={training_mode}",
                ],
                shlex.split(train_args),
            ),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os_environ_cuda,
            },
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
        else:
            return True
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    return False


def run_eval(
    ckpt_path, dataset, num_query_sample, num_shot, model_name, gpu_id, eval_args, eval_mode
):
    try:
        # Determine CUDA_VISIBLE_DEVICES based on the assigned gpu_id
        # If gpu_id is 'cpu', set CUDA_VISIBLE_DEVICES to an empty string
        if gpu_id == "cpu":
            os_environ_cuda = ""
        else:
            os_environ_cuda = str(gpu_id)

        process = subprocess.Popen(
            [sys.executable, "eval.py"]
            + merge_args(
                [
                    f"ckpt_path={ckpt_path or 'null'}",
                    f"model_name={model_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                    f"eval_mode={eval_mode}",
                ],
                shlex.split(eval_args),
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os_environ_cuda},
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
            print(
                f"Evaluation failed on GPU {gpu_id} for runname: {ckpt_path}, dataset: {dataset}"
            )
        else:
            return True
    except Exception as e:
        print(f"An error occurred during evaluation: {e}", file=sys.stderr)
    return False


def run_analyze(runname, dataset, num_query_sample, num_shot, model_name, analyze_args):
    if "icl" in runname:
        # runname-model-dataset
        expand_runname = f"{runname}-{model_name}-{dataset}"
    else:
        # runname-model-dataset-training_samples-num_shot
        expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
    try:
        subprocess.run(
            ["python", "analyze.py"]
            + merge_args(
                [
                    f"model_name={model_name}",
                    f"record_dir={os.path.join(paths.result_dir, 'record', expand_runname)}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(analyze_args),
            ),
            env=os.environ,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Analyze failed for runname: {runname}, dataset: {dataset}",
            file=sys.stderr,
        )
        sys.exit(1)


def execute_eval(runname, tasks, model_name, eval_args, devices, requires_memory, eval_mode):
    task_queue = tasks.copy()
    futures = {}

    def get_gpu_id():
        nonlocal futures
        
        avail_devices_str = get_avail_devices(devices)

        if avail_devices_str == "cpu":
            # If CPU mode is explicitly requested and available, always return 'cpu'
            return "cpu"

        available_gpus = [
            gpu_id
            for gpu_id in avail_devices_str.split(",")
            if gpu_id not in futures
        ]
        if not available_gpus:
            # no available GPUs, wait for tasks to finish
            next(as_completed(futures.values())).result()
            available_gpus = [gpu_id for gpu_id, f in futures.items() if f.done()]
            futures = {gpu_id: f for gpu_id, f in futures.items() if not f.done()}

        return available_gpus.pop(0)

    with ThreadPoolExecutor() as executor:

        while task_queue or futures:
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            if "icl" in runname:
                # runname-model-dataset
                expand_runname = f"{runname}-{model_name}-{dataset}"
                gpu_id = get_gpu_id()
                print(
                    f"Assigning task to GPU {gpu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, ICL"
                )
                futures[gpu_id] = executor.submit(
                    run_eval,
                    None,
                    dataset,
                    num_query_sample,
                    num_shot,
                    model_name,
                    gpu_id,
                    eval_args,
                    eval_mode,
                )
            else:
                # runname-model-dataset-training_samples-num_shot
                expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
                ckpt_dir = os.path.join(paths.result_dir, "ckpt", expand_runname)
                if "baseline" in runname and not os.path.exists(ckpt_dir):
                    # This is a non-ICL baseline run and checkpoint directory doesn't exist (as expected)
                    gpu_id = get_gpu_id()
                    print(
                        f"Assigning task to GPU {gpu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, Baseline (No Checkpoints)"
                    )
                    futures[gpu_id] = executor.submit(
                        run_eval,
                        None,  # ckpt_path is None for baseline
                        dataset,
                        num_query_sample,
                        num_shot,
                        model_name,
                        gpu_id,
                        eval_args,
                        eval_mode,
                    )
                elif os.path.exists(ckpt_dir):
                    # This is a trained model evaluation, proceed with checkpoints
                    epoch_ckpts = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)) and d.startswith("epoch-")]
                    if epoch_ckpts:
                        # Find the latest epoch
                        latest_epoch = -1
                        latest_epoch_ckpt_name = None
                        for epoch_ckpt_name in epoch_ckpts:
                            match = re.search(r"epoch-(\d+)", epoch_ckpt_name)
                            if match:
                                epoch_num = int(match.group(1))
                                if epoch_num > latest_epoch:
                                    latest_epoch = epoch_num
                                    latest_epoch_ckpt_name = epoch_ckpt_name
                        
                        if latest_epoch_ckpt_name:
                            ckpt_to_eval = os.path.join(ckpt_dir, latest_epoch_ckpt_name)
                            gpu_id = get_gpu_id()
                            print(
                                f"Assigning task to GPU {gpu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, latest_epoch={latest_epoch}"
                            )
                            futures[gpu_id] = executor.submit(
                                run_eval,
                                ckpt_to_eval,
                                dataset,
                                num_query_sample,
                                num_shot,
                                model_name,
                                gpu_id,
                                eval_args,
                                eval_mode,
                            )
                        else:
                            print(f"No valid epoch checkpoints found in {ckpt_dir}.", file=sys.stderr)
                    else:
                        print(f"Checkpoint directory {ckpt_dir} exists but contains no epoch checkpoints.", file=sys.stderr)
                else:
                    # Checkpoint directory doesn't exist for a non-baseline runname
                    print(
                        f"Checkpoint directory not found for trained model evaluation (runname does not contain 'baseline'): {ckpt_dir}",
                        file=sys.stderr,
                    )
                    continue # Skipping this task


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation tasks.")
    parser.add_argument(
        "-r", "--runname", required=True, help="Name for the current run."
    )
    parser.add_argument(
        "-d", "--datasets", required=True, help="Comma-separated list of datasets."
    )
    parser.add_argument(
        "-m", "--model-name", required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "-q",
        "--num-query-samples",
        default="0",
        help="Comma-separated list of query samples.",
    )
    parser.add_argument(
        "-s", "--num-shots", required=True, help="Comma-separated list of shots."
    )
    parser.add_argument("-t", "--train", action="store_true", help="Enable train mode.")
    parser.add_argument(
        "--training-mode",
        required=False,
        default="TRAIN_STUDENT_ONESHOT",
        help="Training mode for student model (TRAIN_STUDENT_ONESHOT or TRAIN_STUDENT_DIRECT_Q).",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Enable eval mode. If set to true, analyze mode will be also enabled.",
    )
    parser.add_argument(
        "--eval-mode",
        required=False,
        default="EVAL_BASELINE",
        help="Evaluation mode (EVAL_BASELINE, EVAL_WITH_COT_VECTOR_ONESHOT, EVAL_WITH_COT_VECTOR_DIRECT_Q).",
    )
    parser.add_argument(
        "-a", "--analyze", action="store_true", help="Enable analyze mode."
    )
    parser.add_argument(
        "--train-args",
        required=False,
        help="Additional training arguments.",
    )
    parser.add_argument(
        "--eval-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--analyze-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--devices",
        required=False,
        help="Comma-separated list of integers that denotes devices used to train or evaluate.",
    )
    parser.add_argument(
        "--wait-devices-timeout",
        type=int,
        default=0,
        help="Maximum time in minutes to wait for free GPUs. If <= 0, exit immediately if no adequate GPUs are available.",
    )
    parser.add_argument(
        "--requires_memory",
        type=int,
        default=20000,
        help="The minimal cuda memory used to run train or eval, unit MB.",
    )
    parser.add_argument(
        "--wait-n-devices",
        type=int,
        default=1,
        help="Minimum devices are required to start.",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    num_query_samples = args.num_query_samples.split(",")
    num_shots = args.num_shots.split(",")
    runname = args.runname
    model_name = args.model_name
    train_args = getattr(args, "train_args") or ""
    eval_args = getattr(args, "eval_args") or ""
    analyze_args = getattr(args, "analyze_args") or ""
    devices = args.devices
    timeout = args.wait_devices_timeout
    min_devices = max(args.wait_n_devices, 1)
    requires_memory = args.requires_memory
    training_mode = args.training_mode
    eval_mode = args.eval_mode

    if not (args.train | args.eval | args.analyze):
        args.train = args.eval = args.analyze = True

    if (
        not get_avail_devices(devices, requires_memory)  # no even one devices
        or len(get_avail_devices(devices, requires_memory).split(","))
        < min_devices  # less than minimal required devices
    ) and (args.eval or args.train):
        print(f"Cannot find at least {min_devices} devcie(s). Start waiting...")
        for i in range(0, timeout):
            if (
                get_avail_devices(devices, requires_memory)
                and len(get_avail_devices(devices, requires_memory).split(","))
                >= min_devices
            ):
                break
            time.sleep(60)
            print(f"Waited for {min_devices} device(s) for {(i+1)} mins")
        else:
            print(f"Cannot find at least {min_devices} devcie(s). Timeout, exit...")
            return

    if num_query_samples:
        tasks = list(product(datasets, num_query_samples, num_shots))
    else:
        if args.train:
            raise RuntimeError(
                "The option -q/--num_query_samples is required if train mode is enabled."
            )
        tasks = list(product(datasets, num_shots))

    if args.train:
        print("Starting training phase...")
        task_queue = tasks.copy()
        while task_queue:
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            ret = run_train(
                runname,
                dataset,
                num_query_sample,
                num_shot,
                model_name,
                train_args,
                devices,
                training_mode,
            )

            if isinstance(ret, tuple):
                task_queue.append(ret)
            elif not ret:
                return

    if args.eval:
        print("Starting evaluation phase...")
        execute_eval(runname, tasks, model_name, eval_args, devices, requires_memory, eval_mode)

    if args.analyze:
        print("Starting analysis phase...")
        for dataset, num_query_sample, num_shot in tasks:
            run_analyze(
                runname, dataset, num_query_sample, num_shot, model_name, analyze_args
            )


if __name__ == "__main__":
    main()
