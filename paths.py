from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
TESTBED_DIR = PROJECT_ROOT / "testbed"
RESULT_DIR = Path(os.environ.get("COT_MIMIC_RESULT_DIR", str(PROJECT_ROOT / "results")))


# Lowercase aliases match the legacy module contract used across the repo.
project_root = str(PROJECT_ROOT)
src_dir = str(SRC_DIR)
testbed_dir = str(TESTBED_DIR)
result_dir = str(RESULT_DIR)


# Optional model/data roots referenced by some legacy code paths.
# The current Qwen text pipeline only needs `result_dir`, but keeping these
# names available avoids import-time breakage in adjacent scripts.
idefics_9b_path = os.environ.get("COT_MIMIC_IDEFICS_9B_PATH", "/path/to/your/local/idefics-9b/")
idefics2_8b_base_path = os.environ.get("COT_MIMIC_IDEFICS2_8B_BASE_PATH", "/path/to/your/local/idefics2-8b-base/")
karpathy_coco_caption_dir = os.environ.get("COT_MIMIC_KARPATHY_COCO_DIR", "/path/to/your/local/karpathy-coco/")
coco_dir = os.environ.get("COT_MIMIC_COCO_DIR", "/path/to/your/local/coco/")
flickr30k_dir = os.environ.get("COT_MIMIC_FLICKR30K_DIR", "/path/to/your/local/flickr30k/")
flickr30k_images_dir = os.environ.get("COT_MIMIC_FLICKR30K_IMAGES_DIR", "/path/to/your/local/flickr30k-images/")
seed_dir = os.environ.get("COT_MIMIC_SEED_DIR", "/path/to/your/local/seed_bench/")
