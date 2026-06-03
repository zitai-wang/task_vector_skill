from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

local_path = "/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/"

categories = ['algebra', 'counting_and_probability', 'geometry',
              'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

easy_levels = ['Level 2', 'Level 3', 'Level 1']
hard_levels = ['Level 4', 'Level 5']

all_splits_train= []
all_splits_test = []
for cat in categories:
    ds = load_dataset(local_path, cat)
    all_splits_train.append(ds['train'])
    all_splits_test.append(ds['test'])

all_train = concatenate_datasets(all_splits_train)
all_test = concatenate_datasets(all_splits_test)

train_easy = all_train.filter(lambda x: x["level"] in easy_levels)
train_hard = all_train.filter(lambda x: x["level"] in hard_levels)
test_easy  = all_test.filter(lambda x: x["level"] in easy_levels)
test_hard  = all_test.filter(lambda x: x["level"] in hard_levels)

train_easy = train_easy.shuffle(seed=42)
train_hard = train_hard.shuffle(seed=42)
test_easy  = test_easy.shuffle(seed=42)
test_hard  = test_hard.shuffle(seed=42)

train_easy.save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_train_subset_easy")
train_hard.save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_train_subset_hard")
test_easy .save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_test_subset_easy")
test_hard .save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_test_subset_hard")


# all_splits_train_easy = []
# all_splits_train_hard = []
# all_splits_test_easy = []
# all_splits_test_hard = []
# for cat in categories:
#     ds = load_dataset(local_path, cat)
#     for sample in ds['train']:
#         if sample["level"] in easy_levels:
#             all_splits_train_easy.append(Dataset.from_dict(sample))
#         elif sample["level"] in hard_levels:
#             all_splits_train_hard.append(Dataset.from_dict(sample))
#         else:
#             continue
#     for sample in ds['test']:
#         if sample["level"] in easy_levels:
#             all_splits_test_easy.append(sample)
#         elif sample["level"] in hard_levels:
#             all_splits_test_hard.append(sample)
#         else:
#             continue

train_easy_full = concatenate_datasets(all_splits_train_easy)
train_hard_full = concatenate_datasets(all_splits_train_hard)
test_easy_full = concatenate_datasets(all_splits_test_easy)
test_hard_full = concatenate_datasets(all_splits_test_hard)

train_df_easy = train_easy_full.to_pandas()
train_df_hard = train_hard_full.to_pandas()
test_df_easy = test_easy_full.to_pandas()
test_df_hard = test_hard_full.to_pandas()

print("Train类别分布:\n", train_df_easy['type'].value_counts())
print("Test类别分布:\n", test_df_easy['type'].value_counts())

# 从原始 train 中抽 5000 条
train_sample_easy, _ = train_test_split(
    train_df_easy,
    train_size=5000,
    stratify=train_df_easy['type'],
    random_state=42
)

train_sample_hard, _ = train_test_split(
    train_df_hard,
    train_size=5000,
    stratify=train_df_hard['type'],
    random_state=42
)

# 从原始 test 中抽 1000 条
test_sample_easy, _ = train_test_split(
    test_df_easy,
    train_size=1000,
    stratify=test_df_easy['type'],
    random_state=42
)

test_sample_hard, _ = train_test_split(
    test_df_hard,
    train_size=1000,
    stratify=test_df_hard['type'],
    random_state=42
)

train_ds = Dataset.from_pandas(train_sample)
test_ds = Dataset.from_pandas(test_sample)

train_ds.save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_train_5000_subset")
test_ds.save_to_disk("/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/Datasets/MATH/math_test_1000_subset")


# new = Dataset.load_from_disk(path)