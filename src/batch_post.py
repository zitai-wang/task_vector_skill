import os
import re
import sys
import subprocess

INPUT_DIR = "/data1/wzy/cot-mimic/results/record/vl_gsm8k_base_vector"
POSTPROCESS_SCRIPT = "/data1/wzy/cot-mimic/src/post_mathvista.py"

# FILE_PATTERN = re.compile(r"^licv_layers_(\d+)_direct_q_[0-9.]+\.json$")
FILE_PATTERN = re.compile(r"^use_base_vector_layers_(\d+)_direct_q_[0-9.]+\.json$")
ACC_PATTERN = re.compile(r"处理后准确率[:：]\s*([0-9.]+)%")


def find_input_files(input_dir: str):
    files = []
    for name in os.listdir(input_dir):
        m = FILE_PATTERN.match(name)
        if m:
            layer_id = int(m.group(1))
            full_path = os.path.join(input_dir, name)
            files.append((layer_id, full_path))
    files.sort(key=lambda x: x[0])
    return files


def run_one_file(input_path: str):
    cmd = [sys.executable, POSTPROCESS_SCRIPT, input_path, "--no-save"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"后处理失败\n"
            f"文件: {input_path}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    m = ACC_PATTERN.search(stdout)
    if not m:
        raise ValueError(
            f"没有解析到处理后准确率\n"
            f"文件: {input_path}\n"
            f"stdout:\n{stdout}"
        )

    acc = float(m.group(1))
    return acc, stdout


def main():
    files = find_input_files(INPUT_DIR)

    if not files:
        print(f"没有找到符合规则的文件: {INPUT_DIR}")
        return

    print("=" * 100)
    print(f"共找到 {len(files)} 个文件，开始批量后处理")
    print("=" * 100)

    results = []

    for layer_id, input_path in files:
        file_name = os.path.basename(input_path)
        print(f"\n[Layer {layer_id:02d}] 正在处理 {file_name}")

        try:
            acc, _ = run_one_file(input_path)
            results.append((layer_id, acc))
            print(f"[Layer {layer_id:02d}] post accuracy = {acc:.4f}%")
        except Exception as e:
            results.append((layer_id, None))
            print(f"[Layer {layer_id:02d}] FAILED")
            print(e)

    print("\n" + "=" * 100)
    print("所有 post 准确率如下：")
    print("=" * 100)

    for layer_id, acc in results:
        if acc is None:
            print(f"layer {layer_id:02d}: FAILED")
        else:
            print(f"layer {layer_id:02d}: {acc:.4f}%")

    valid_results = [(layer_id, acc) for layer_id, acc in results if acc is not None]
    if valid_results:
        best_layer, best_acc = max(valid_results, key=lambda x: x[1])
        print("=" * 100)
        print(f"最佳结果: layer {best_layer:02d} -> {best_acc:.4f}%")

    print("=" * 100)


if __name__ == "__main__":
    main()
