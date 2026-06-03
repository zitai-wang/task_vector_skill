# Self-CoT Training Extension for MimIC

这个扩展为MimIC框架添加了Self-CoT训练功能，通过使用模型自己生成的CoT推理过程来改善训练效果。

## 背景

原始的MimIC训练使用Ground Truth (GT) CoT，但发现Qwen模型对GT CoT的loss较大，说明模型不太熟悉GT CoT的思路。因此，我们提出使用模型自己生成的CoT（Self-CoT）来训练，只选择模型能正确回答的样本。

## 工作流程

### 1. 生成Self-CoT数据

首先需要生成Self-CoT数据：

```bash
cd src
python generate_self_cot.py
```

这个脚本会：
- 加载GSM8K训练集
- 使用Qwen模型生成CoT推理过程
- 评估生成答案的正确性
- 保存所有结果和仅正确的结果

配置文件：`config/generate_self_cot.yaml`

### 2. 使用Self-CoT训练

使用Self-CoT数据进行训练：

```bash
cd src
python train.py --config-name train_self_cot.yaml
```

或者使用pipeline：

```bash
python pipeline.py -r self_cot_run -d gsm8k -m qwen2.5-math-7b-instruct -q 5000 -s 1 -t --train-args "config_name=train_self_cot"
```

## 新增的训练模式

### DatasetState枚举扩展

```python
# 新增的Self-CoT训练模式
TRAIN_TEACHER_SELF_COT = "TRAIN_TEACHER_SELF_COT"  # Teacher使用Self-CoT而不是GT-CoT
TRAIN_STUDENT_ONESHOT_SELF_COT = "TRAIN_STUDENT_ONESHOT_SELF_COT"  # 学生模型使用Self-CoT teacher
TRAIN_STUDENT_DIRECT_Q_SELF_COT = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"  # 学生模型使用Self-CoT teacher
```

### 配置文件扩展

在数据配置中添加Self-CoT相关参数：

```yaml
data:
  # Self-CoT configuration
  use_self_cot: True  # 启用Self-CoT训练
  self_cot_path: "/path/to/self_cot_data_correct_only.jsonl"  # Self-CoT数据路径
  
  # Training mode
  training_mode: TRAIN_STUDENT_DIRECT_Q_SELF_COT  # 使用Self-CoT的训练模式
```

## 数据格式

Self-CoT数据文件格式（JSONL）：

```json
{
  "question": "Janet's dogs eat 2 pounds of food each week. How many pounds of food do her dogs eat in 8 weeks?",
  "gt_answer": "Janet's dogs eat 2 pounds of food each week.\nTo find out how many pounds they eat in 8 weeks, we multiply the weekly amount by the number of weeks.\n2 pounds/week × 8 weeks = 16 pounds\n#### 16",
  "gt_numerical": "16",
  "self_cot": "Let me solve this step by step:\n\n1) Janet's dogs eat 2 pounds of food each week\n2) We need to find out how much they eat in 8 weeks\n3) To do this, we multiply the weekly amount by the number of weeks\n4) 2 pounds × 8 weeks = 16 pounds\n\nTherefore, Janet's dogs eat \boxed{16} pounds of food in 8 weeks.",
  "predicted_answer": "16",
  "is_correct": true
}
```

## 主要改进

### 1. 数据过滤
- 只使用模型能正确回答的样本进行训练
- 自动过滤掉错误样本，提高训练质量

### 2. Teacher输入改进
- 原始：Teacher输入 = Q + GT-CoT + Answer
- 改进：Teacher输入 = Q + Self-CoT + Answer

### 3. 训练流程
1. **预生成阶段**：生成Self-CoT数据并筛选正确样本
2. **训练阶段**：使用Self-CoT数据进行MimIC训练
3. **评估阶段**：使用原有评估流程

## 使用示例

### 完整工作流程

```bash
# 1. 生成Self-CoT数据
cd src
python generate_self_cot.py

# 2. 检查生成的数据
ls -la /home/share/pyz/dataset/gsm8k/self_cot_data*.jsonl

# 3. 使用Self-CoT训练
python train.py --config-name train_self_cot.yaml

# 4. 评估训练结果
python eval.py --config-name eval.yaml ckpt_path=/path/to/checkpoint
```

### 配置参数说明

#### generate_self_cot.yaml
- `max_samples`: 最大处理样本数（0表示全部）
- `output_path`: 输出文件路径
- `generation_args`: 生成参数（温度、top_p等）

#### train_self_cot.yaml
- `use_self_cot`: 是否启用Self-CoT
- `self_cot_path`: Self-CoT数据文件路径
- `training_mode`: 训练模式选择

## 优势

1. **更好的对齐**：使用模型自己生成的CoT，减少对齐难度
2. **质量保证**：只使用正确样本，避免错误推理的影响
3. **向后兼容**：不影响原有的GT-CoT训练流程
4. **灵活配置**：可以轻松切换GT-CoT和Self-CoT训练

## 注意事项

1. **数据路径**：确保Self-CoT数据文件路径正确
2. **内存使用**：Self-CoT生成可能需要较多内存
3. **时间成本**：预生成Self-CoT数据需要额外时间
4. **模型一致性**：建议使用相同的模型进行Self-CoT生成和训练

## 扩展性

这个设计可以轻松扩展到其他数据集和模型：
1. 修改`generate_self_cot.py`中的数据集加载
2. 调整答案提取和正确性判断逻辑
3. 更新配置文件中的路径和参数 