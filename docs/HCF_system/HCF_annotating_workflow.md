# HCF 标注工作流文档

本文档记录手部左/右手二分类器 (HCF) 系统每个操作步骤的命令、操作内容和底层原理。

## 概述

HCF 系统训练一个 MobileNetV3-Small 模型，对 Hand ROI 图像进行 Left（左手，标签 0）/ Right（右手，标签 1）二分类。训练好的模型导出为 ONNX 格式，用于对 CVAT 自动标注 XML 文件进行重新标注，将其中的 `unknown_handedness` 标签替换为预测的 handedness 标签。

流水线包含以下五个阶段：

1. 数据集统计
2. 训练
3. 评估
4. ONNX 导出
5. CVAT 标签导出测试

---

## 阶段 1：数据集统计

### 命令

```bash
python scripts/dataset_stats.py --config configs/hand_classifier.yaml
```

### 输入

- `configs/hand_classifier.yaml` 中 `data.train_sources` 指定的数据来源目录（位于服务器端）。
- 每个数据来源目录包含 `images/*.png` 和一个 CVAT XML 标注文件。

### 操作

- 遍历 `data.train_sources` 中列出的所有数据来源目录。
- 对每个数据来源，解析 CVAT XML 文件（优先选择 `cvat_reviewed.xml`，其次 `cvat_autolabel.xml`）。
- 统计每个数据来源中 Left、Right 样本数，以及被排除的样本（ignore_for_training、no_hand、unknown_handedness）。

### 输出

- 在终端打印每个数据来源的标签分布表（Left/Right 数量及比例）。

### 原理

在训练前验证数据完整性，了解类别分布情况。约 2.75:1 的 Left:Right 不平衡比例是训练时使用平衡类别权重的依据。

---

## 阶段 2：训练

### 命令

```bash
python scripts/train.py --config configs/hand_classifier.yaml
```

### 输入

- `configs/hand_classifier.yaml` - 包含所有超参数。
- `autodl-tmp/DatasetFab/HCFTrainSource/*` 下的数据来源目录。

### 操作

1. **数据收集**：解析所有数据来源目录，收集带有 Left/Right 标签的样本。
2. **数据集划分**：当前配置指定了 2 个固定验证集（`complex-near-bright-random-test-s01-peak` 和 `complex-mid-bright-random-val-s01-peak`），其余 6 个数据来源全部用作训练集。程序自动从训练集中排除验证集来源，避免数据泄漏。若 `val_sources` 为空，则回退到在每个数据来源内部按标签进行 90/10 分层抽样的划分方式。当前不划分测试集。
3. **模型构建**：构建 MobileNetV3-Small 模型，第一层卷积的 RGB 三通道预训练权值平均为单通道，分类头替换为二分类输出。
4. **训练循环**：
   - 损失函数：`CrossEntropyLoss`，使用平衡类别权重（逆频率）。
   - 优化器：`AdamW`，采用差异化学习率（backbone 使用基础 lr，分类头使用 lr×10）。
   - 学习率调度：`CosineAnnealingLR` + 线性预热（5 个 epoch）。
   - 启用自动混合精度（AMP，仅 GPU）。
   - 数据增强：水平翻转（同步交换 0↔1 标签）、随机仿射变换、色彩抖动、随机擦除。
5. **早停机制**：监控验证损失（val_loss），patience=15 个 epoch。

### 输出

- `outputs/splits.json` - 训练集和验证集的划分元数据。
- `outputs/checkpoints/best.pth` - 最佳模型检查点（按 val_loss 最低）。
- `outputs/checkpoints/last.pth` - 最新模型检查点。
- `outputs/train/metrics.jsonl` - 每个 epoch 的训练指标。

### 参数调整原理

- `training.batch_size: 64`：对于 256×256 单通道图像，RTX 3090 24GB 显存可轻松容纳。
- `training.learning_rate: 0.0001`：微调预训练模型的标准学习率。
- `training.head_lr_multiplier: 10`：新分类头未经过预训练，需要比 backbone 更快的收敛速度。
- `training.warmup_epochs: 5`：预热有助于在差异化学习率配置下稳定早期训练。
- `training.class_weights: "balanced"`：补偿约 2.75:1 的 Left:Right 类别不平衡。
- `augmentation.horizontal_flip_prob: 0.5`：水平翻转后必须同步交换 0↔1 标签，保证标签正确性。
- `augmentation.rotation_degrees: 10`：小角度旋转可模拟手部姿态的自然变化。

---

## 阶段 3：评估

### 命令

```bash
python scripts/evaluate.py --config configs/hand_classifier.yaml [--checkpoint outputs/checkpoints/best.pth]
```

### 输入

- 训练好的模型检查点。
- 验证集（由划分产生或来自 `data.val_sources`）。

### 操作

1. 加载最佳检查点。
2. 对验证集进行推理（不做数据增强，仅做归一化）。
3. 计算指标：准确率（Accuracy）、各类别精确率/召回率/F1值、混淆矩阵、ROC-AUC。
4. 计算每个数据来源的单独准确率。

### 输出

- 终端输出：所有评估指标。
- `outputs/eval/val_metrics.json` - 完整评估指标 JSON 文件。
- 若指定了测试来源，还会输出 `outputs/eval/test_metrics.json`。

---

## 阶段 4：ONNX 导出

### 命令

```bash
python scripts/export_onnx.py --config configs/hand_classifier.yaml [--checkpoint outputs/checkpoints/best.pth] [-o outputs/model.onnx]
```

### 输入

- 训练好的模型检查点。

### 操作

1. 加载模型和检查点。
2. 将模型移至 CPU。
3. 使用 `torch.onnx.export` 导出（opset=13，动态 batch，开启常量折叠）。
4. 验证：使用 `onnx.checker` 校验 ONNX 模型结构，使用 `onnxruntime` 进行推理并检查输出形状。

### 输出

- `outputs/model.onnx` - 支持动态 batch 的 ONNX 模型文件。

### 原理

- opset=13 是广泛支持的 ONNX 算子集版本，覆盖 MobileNetV3 的所有算子。
- 动态 batch 允许推理阶段处理任意批量大小。
- 验证步骤确保导出后的 ONNX 模型在部署前是有效的。

---

## 阶段 5：CVAT 标签导出测试

### 命令

```bash
python scripts/cvat_label_test.py --config configs/hand_classifier.yaml [--checkpoint outputs/model.onnx] [--xml <path>]
```

### 输入

- ONNX 模型文件。
- CVAT 自动标注 XML 文件（`cvat_autolabel.xml`），其中所有 `<tag>` 标签均为 `unknown_handedness`。
- 图像目录（默认：XML 文件所在目录的 `../images/`）。

### 操作

1. 使用 onnxruntime 加载 ONNX 模型。
2. 通过正则表达式匹配 `<image>...</image>` 块，解析 XML。
3. 对每张图像运行推理，得到预测结果 Left（0）或 Right（1）。
4. 在原标签位置将 `label="unknown_handedness"` 替换为 `label="Left"` 或 `label="Right"`。
5. 所有空格、骨架数据、关键点坐标、格式风格均原样保留（基于字节保留的正则替换）。
6. 如果存在 `cvat_reviewed.xml`（金标准），则计算一致性率。

### 输出

- 重新标注后的 XML 文件（默认：`outputs/cvat_relabeled.xml`）。
- 终端输出：总图像数、Left 数、Right 数、错误数、与金标准的一致性率（如有）。

### 原理

采用字节保留的正则替换方案（而非 XML 序列化方案），确保输出文件与原始格式尽可能一致。XML 序列化会不可避免地改变空格、引号风格、自闭合标签约定等细节。
