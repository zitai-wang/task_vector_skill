#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path("/data1/wzy/cot-mimic")
RESULTS_DIR = ROOT / "results"
CKPT_ROOT = RESULTS_DIR / "ckpt"
RECORD_ROOT = RESULTS_DIR / "record"
LOG_ROOT = ROOT / "scripts" / "mmmu_learnable_layer_sweep"
DEFAULT_PYTHON = "/home/wzy/anaconda3/envs/licv/bin/python"
DEFAULT_LAYERS = list(range(28))
EXPECTED_SAMPLES = 900
DEFAULT_TRAIN_FREE_MB = 22000
DEFAULT_EVAL_FREE_MB = 22000
RUN_PREFIX = "mmmu-qwen2.5-vl-mimic"
ACTIVE_LOG_ROOT = LOG_ROOT


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return DEFAULT_LAYERS

    layers: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {chunk}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(chunk))

    parsed = sorted(layers)
    for layer in parsed:
        if layer < 0 or layer > 27:
            raise ValueError(f"Layer index out of range [0, 27]: {layer}")
    return parsed


def runname_for_layer(layer: int) -> str:
    return f"{RUN_PREFIX}-l{layer}"


def expand_runname(layer: int) -> str:
    return f"{runname_for_layer(layer)}-qwen2.5-vl-7b-instruct-mmmu-500"


def ckpt_dir_for_layer(layer: int) -> Path:
    return CKPT_ROOT / expand_runname(layer)


def latest_epoch_dir(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None

    epoch_dirs = []
    for child in ckpt_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"epoch-(\d+)", child.name)
        if match:
            epoch_dirs.append((int(match.group(1)), child))
    if not epoch_dirs:
        return None
    epoch_dirs.sort(key=lambda item: item[0])
    return epoch_dirs[-1][1]


def record_path_for_layer(layer: int) -> Path | None:
    epoch_dir = latest_epoch_dir(ckpt_dir_for_layer(layer))
    if epoch_dir is None:
        return None
    epoch_match = re.fullmatch(r"epoch-(\d+)", epoch_dir.name)
    if epoch_match is None:
        return None
    return RECORD_ROOT / expand_runname(layer) / f"epoch-{epoch_match.group(1)}.json"


def read_eval_result(record_path: Path | None) -> dict | None:
    if record_path is None or not record_path.exists() or record_path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(record_path.read_text())
    except Exception:
        return None

    accuracy = payload.get("eval_result", {}).get("accuracy")
    records = payload.get("records", [])
    if accuracy is None or len(records) != EXPECTED_SAMPLES:
        return None

    return {
        "accuracy": accuracy,
        "num_records": len(records),
        "record_path": str(record_path),
    }


def ensure_log_dir() -> None:
    ACTIVE_LOG_ROOT.mkdir(parents=True, exist_ok=True)


def truncate_layer_logs(layers: list[int], do_train: bool, do_eval: bool) -> None:
    ensure_log_dir()
    for layer in layers:
        if do_train:
            train_log = ACTIVE_LOG_ROOT / f"layer_{layer:02d}_train.log"
            if train_log.exists():
                train_log.unlink()
        if do_eval:
            eval_log = ACTIVE_LOG_ROOT / f"layer_{layer:02d}_eval.log"
            if eval_log.exists():
                eval_log.unlink()


def build_env(gpu: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    return env


def query_gpu_memory() -> dict[str, dict[str, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}
    rows: dict[str, dict[str, int]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows[parts[0]] = {
            "used_mb": int(parts[1]),
            "free_mb": int(parts[2]),
            "util": int(parts[3]),
        }
    return rows


def wait_for_gpu_capacity(
    gpu: str,
    min_free_mb: int,
    stream_label: str,
    poll_seconds: int = 20,
) -> None:
    while True:
        stats = query_gpu_memory().get(gpu)
        if stats is None:
            print(
                f"[{stream_label}] GPU state unavailable from nvidia-smi, "
                f"skipping capacity gate for GPU {gpu}.",
                flush=True,
            )
            return

        if stats["free_mb"] >= min_free_mb:
            print(
                f"[{stream_label}] GPU {gpu} ready: "
                f"free={stats['free_mb']}MB used={stats['used_mb']}MB util={stats['util']}%",
                flush=True,
            )
            return

        print(
            f"[{stream_label}] GPU {gpu} busy, waiting: "
            f"free={stats['free_mb']}MB < required={min_free_mb}MB "
            f"(used={stats['used_mb']}MB util={stats['util']}%)",
            flush=True,
        )
        time.sleep(poll_seconds)


def run_command(
    cmd: list[str],
    log_path: Path,
    gpu: str,
    max_retries: int,
    stream_label: str,
    min_free_mb: int,
) -> None:
    ensure_log_dir()
    last_return_code = None
    for attempt in range(1, max_retries + 1):
        wait_for_gpu_capacity(gpu, min_free_mb, stream_label)
        with log_path.open("a", encoding="utf-8") as log_file:
            header = (
                f"\n[START {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"GPU={gpu} attempt={attempt}/{max_retries}\n"
            )
            command_line = "COMMAND: " + " ".join(cmd) + "\n\n"
            log_file.write(header)
            log_file.write(command_line)
            log_file.flush()
            print(f"[{stream_label}] {header.strip()}", flush=True)
            print(f"[{stream_label}] {command_line.strip()}", flush=True)
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=build_env(gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(f"[{stream_label}] {line}", end="", flush=True)
            last_return_code = process.wait()
            footer = (
                f"\n[END {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"returncode={last_return_code}\n"
            )
            log_file.write(footer)
            log_file.flush()
            print(f"[{stream_label}] {footer.strip()}", flush=True)

        if last_return_code == 0:
            return

        time.sleep(5)

    raise RuntimeError(f"Command failed with exit code {last_return_code}: {' '.join(cmd)}")


def build_train_cmd(python_bin: str, layer: int, extra_train_args: list[str]) -> list[str]:
    cmd = [
        python_bin,
        str(ROOT / "src" / "train.py"),
        "--config-name",
        "train_mmmu_qwenvl",
        f"runname={runname_for_layer(layer)}",
        f"only_shift_at_layer={layer}",
        "layer_train_mode=single_layer",
        f"shift_layers_combinations=[[{layer}]]",
    ]
    cmd.extend(extra_train_args)
    return cmd


def build_eval_cmd(python_bin: str, layer: int, ckpt_path: Path) -> list[str]:
    return [
        python_bin,
        str(ROOT / "src" / "eval_mathvision.py"),
        "--config-name",
        "eval_mmmu",
        "model_name=qwen2.5-vl-7b-instruct",
        "data.name=mmmu",
        "data.source=/data/share/MMMU",
        "data.query_split=validation",
        "data.support_split=dev",
        "data.subsets=null",
        f"data.num_query_samples={EXPECTED_SAMPLES}",
        "data.num_shot=0",
        "batch_size=1",
        "devices=0",
        "resume=False",
        "eval_mode=EVAL_WITH_COT_VECTOR_DIRECT_Q",
        "use_extracted_cot_vector=False",
        "generation_args.max_new_tokens=1024",
        f"ckpt_path={ckpt_path}",
        f"only_shift_at_layer={layer}",
        "+encoder=mimic",
        "+peft=mimic",
    ]


def write_summary(layers: list[int]) -> Path:
    ensure_log_dir()
    summary_rows = []
    for layer in layers:
        ckpt_dir = ckpt_dir_for_layer(layer)
        epoch_dir = latest_epoch_dir(ckpt_dir)
        record_path = record_path_for_layer(layer)
        result = read_eval_result(record_path)
        summary_rows.append(
            {
                "layer": layer,
                "runname": expand_runname(layer),
                "ckpt_dir": str(ckpt_dir),
                "latest_epoch_dir": str(epoch_dir) if epoch_dir else None,
                "record_path": str(record_path) if record_path else None,
                "accuracy": result["accuracy"] if result else None,
                "num_records": result["num_records"] if result else None,
            }
        )

    summary_rows.sort(
        key=lambda item: (-1 if item["accuracy"] is None else 0, -(item["accuracy"] or -1), item["layer"])
    )
    summary_path = ACTIVE_LOG_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def print_summary(layers: list[int]) -> None:
    rows = []
    for layer in layers:
        result = read_eval_result(record_path_for_layer(layer))
        rows.append((layer, None if result is None else result["accuracy"]))

    rows.sort(key=lambda item: (-1 if item[1] is None else 0, -(item[1] or -1), item[0]))
    print("\nCurrent MMMU learnable layer sweep summary:")
    for layer, acc in rows:
        if acc is None:
            print(f"  layer {layer:02d}: pending")
        else:
            print(f"  layer {layer:02d}: {acc:.4f}")

    complete_rows = [(layer, acc) for layer, acc in rows if acc is not None]
    if complete_rows:
        best_layer, best_acc = complete_rows[0]
        print(f"\nBest completed layer so far: layer {best_layer} with accuracy {best_acc:.4f}")


def worker(
    gpu: str,
    task_queue: "queue.Queue[int]",
    python_bin: str,
    do_train: bool,
    do_eval: bool,
    max_retries: int,
    train_min_free_mb: int,
    eval_min_free_mb: int,
    extra_train_args: list[str],
    lock: threading.Lock,
    failures: list[dict],
) -> None:
    while True:
        try:
            layer = task_queue.get_nowait()
        except queue.Empty:
            return

        try:
            with lock:
                print(f"[GPU {gpu}] Start layer {layer}")

            epoch_dir = latest_epoch_dir(ckpt_dir_for_layer(layer))
            if do_train and epoch_dir is None:
                train_log = ACTIVE_LOG_ROOT / f"layer_{layer:02d}_train.log"
                run_command(
                    build_train_cmd(python_bin, layer, extra_train_args),
                    train_log,
                    gpu,
                    max_retries,
                    f"layer{layer:02d}-train-gpu{gpu}",
                    train_min_free_mb,
                )
                epoch_dir = latest_epoch_dir(ckpt_dir_for_layer(layer))
                if epoch_dir is None:
                    raise RuntimeError(f"Layer {layer} training finished but no checkpoint directory was found.")
            elif do_train:
                with lock:
                    print(f"[GPU {gpu}] Skip train layer {layer}, checkpoint already exists: {epoch_dir}")

            existing_result = read_eval_result(record_path_for_layer(layer))
            if do_eval and existing_result is None:
                if epoch_dir is None:
                    raise RuntimeError(f"Layer {layer} has no checkpoint to evaluate.")
                eval_log = ACTIVE_LOG_ROOT / f"layer_{layer:02d}_eval.log"
                run_command(
                    build_eval_cmd(python_bin, layer, epoch_dir),
                    eval_log,
                    gpu,
                    max_retries,
                    f"layer{layer:02d}-eval-gpu{gpu}",
                    eval_min_free_mb,
                )
                existing_result = read_eval_result(record_path_for_layer(layer))
                if existing_result is None:
                    raise RuntimeError(f"Layer {layer} evaluation finished but no valid result file was found.")
            elif do_eval:
                with lock:
                    print(
                        f"[GPU {gpu}] Skip eval layer {layer}, existing accuracy="
                        f"{existing_result['accuracy']:.4f}"
                    )

            with lock:
                if existing_result is None:
                    print(f"[GPU {gpu}] Finished layer {layer} without eval result.")
                else:
                    print(f"[GPU {gpu}] Finished layer {layer}, accuracy={existing_result['accuracy']:.4f}")
        except Exception as exc:
            failures.append({"layer": layer, "gpu": gpu, "error": str(exc)})
            with lock:
                print(f"[GPU {gpu}] Layer {layer} failed: {exc}")
        finally:
            task_queue.task_done()


def is_layer_complete(layer: int, do_train: bool, do_eval: bool) -> bool:
    if do_eval:
        return read_eval_result(record_path_for_layer(layer)) is not None
    if do_train:
        return latest_epoch_dir(ckpt_dir_for_layer(layer)) is not None
    return True


def main() -> None:
    global RUN_PREFIX, ACTIVE_LOG_ROOT
    parser = argparse.ArgumentParser(description="Train and evaluate learnable MMMU CoT vectors layer by layer.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON, help="Python executable used for train/eval subprocesses.")
    parser.add_argument("--gpus", default="2", help="Comma-separated GPU ids. Each GPU runs one layer at a time.")
    parser.add_argument("--layers", default="0-27", help="Layer list, e.g. '0-27' or '8,9,10'.")
    parser.add_argument("--run-prefix", default=RUN_PREFIX, help="Runname prefix. Final runname becomes '<run-prefix>-l{layer}'.")
    parser.add_argument("--log-dir", default=str(LOG_ROOT), help="Directory for per-layer logs and summary.")
    parser.add_argument("--train-arg", action="append", default=[], help="Extra Hydra override passed to train.py. Repeatable.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run evaluation/summarization.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and only run training/summarization.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per train/eval command.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Re-run pending layers until complete or rounds exhausted.")
    parser.add_argument("--train-min-free-mb", type=int, default=DEFAULT_TRAIN_FREE_MB, help="Minimum free GPU memory before launching a train job.")
    parser.add_argument("--eval-min-free-mb", type=int, default=DEFAULT_EVAL_FREE_MB, help="Minimum free GPU memory before launching an eval job.")
    parser.add_argument("--truncate-layer-logs", action="store_true", help="Delete existing per-layer train/eval logs before starting.")
    args = parser.parse_args()
    RUN_PREFIX = args.run_prefix
    ACTIVE_LOG_ROOT = Path(args.log_dir)

    layers = parse_layers(args.layers)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required.")

    do_train = not args.skip_train
    do_eval = not args.skip_eval
    if not do_train and not do_eval:
        summary_path = write_summary(layers)
        print_summary(layers)
        print(f"\nSummary written to {summary_path}")
        return

    ensure_log_dir()
    if args.truncate_layer_logs:
        truncate_layer_logs(layers, do_train, do_eval)
    print(f"Launching MMMU learnable sweep for layers: {layers}")
    print(f"Using GPUs: {gpus}")
    print(f"Run prefix: {RUN_PREFIX}")
    print(f"Extra train args: {args.train_arg}")
    print(f"Logs will be written to: {ACTIVE_LOG_ROOT}")

    round_idx = 1
    while round_idx <= args.max_rounds:
        pending_layers = [layer for layer in layers if not is_layer_complete(layer, do_train, do_eval)]
        if not pending_layers:
            break

        print(f"\nRound {round_idx}: pending layers = {pending_layers}")
        task_queue = queue.Queue()
        for layer in pending_layers:
            task_queue.put(layer)

        lock = threading.Lock()
        failures: list[dict] = []
        threads = []
        for gpu in gpus:
            thread = threading.Thread(
                target=worker,
                args=(
                    gpu,
                    task_queue,
                    args.python_bin,
                    do_train,
                    do_eval,
                    args.max_retries,
                    args.train_min_free_mb,
                    args.eval_min_free_mb,
                    args.train_arg,
                    lock,
                    failures,
                ),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if failures:
            print(f"Round {round_idx} failures: {json.dumps(failures, ensure_ascii=False)}")

        write_summary(layers)
        remaining = [layer for layer in layers if not is_layer_complete(layer, do_train, do_eval)]
        if not remaining:
            break
        print(f"After round {round_idx}, remaining pending layers: {remaining}")
        round_idx += 1

    summary_path = write_summary(layers)
    print_summary(layers)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
