import os
import importlib.util
import inspect

from dataset_utils.mmlu import MMLUProDataset
from .strategyqa import StrategyQADataset
from .interface import DatasetBase
from .gsm8k import GSM8KDataset
from .math_dataset import MATHDataset
from .mmmu import MMMUDataset
from .mathvision import MathVisionDataset
from .mathvista import MathVistaDataset
from .commonsenceqa import CommonsenseQADataset
from .scienceqa import ScienceQADataset

# a mapping from dataset name to Dataset class
dataset_mapping = {
    "gsm8k": GSM8KDataset,
    "mmlu": MMLUProDataset,
    "math_dataset": MATHDataset,
    "mmmu": MMMUDataset,
    "mathvision": MathVisionDataset,
    "mathvista": MathVistaDataset,
    "commonsenseqa": CommonsenseQADataset,
    "strategyqa": StrategyQADataset,
    "scienceqa": ScienceQADataset

}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["interface.py", "__init__.py", "gsm8k.py", "mmlu.py","math_dataset.py", "mmmu.py", "caption.py", "mme.py", "seed_bench.py", "vqa.py", "mathvista.py", "mathvision.py","commonsenceqa.py","strategyqa.py","scienceqa.py"]:
        module_name = filename[:-3]
        filepath = os.path.join(os.path.dirname(__file__), filename)

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        dataset_classes = [
            cls for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls is not DatasetBase
            and cls.__module__ == module.__name__
            and issubclass(cls, DatasetBase)
        ]

        for dataset_class in dataset_classes:
            for dataset in dataset_class.support_datasets:
                dataset_mapping[dataset] = dataset_class
            
__all__ = ["dataset_mapping", "DatasetBase"]
