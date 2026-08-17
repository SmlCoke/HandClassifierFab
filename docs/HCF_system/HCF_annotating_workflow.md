# HCF 标注工作流文档

本文档记录手部左/右手二分类器 (HCF) 系统每个操作步骤的命令、操作内容和底层原理。

## 概述

HCF 系统训练一个双头模型，共享 backbone，同时进行两项分类任务：

1. **handedness**：Left（左手，标签 0）/ Right（右手，标签 1）
2. **hand_presence**：no_hand（无手，标签 0）/ has_hand（有手，标签 1）

模型输出为 dict：`{"handedness": Tensor, "hand_presence": Tensor}`。训练好的模型导出为 ONNX 格式（包含两个输出头），用于对 CVAT 自动标注 XML 文件进行重新标注。

模型分为两个系列（通过配置 `model.version` 选择，输入输出格式完全一致）：

- **v1.0 系列**（`models/v1/`）：原始 MobileNetV3-Small/Large 双头模型（约 150 万 / 420 万参数）。
- **v2.0 系列**（`models/v2/`）：为提升精度而设计的大参数量模型，牺牲延迟换取精度（不参与板端部署，仅用于数据自动化标注）。包含标准卷积 CNN（`v2_convnet_s/l`）、多分支卷积（`v2_multibranch`）、CNN+Transformer 混合（`v2_hybrid_s/l`）、以及 ImageNet 预训练迁移主干（`v2_resnet50`、`v2_convnext_tiny`、`v2_efficientnet_v2_s`、`v2_vit_b16`）。

流水线包含以下六个阶段：

1. 数据集核验
2. 数据集统计
3. 训练
4. 评估
5. ONNX 导出
6. CVAT 标签导出测试

---

## 阶段 1：数据集核验

### 命令

```bash
python scripts/verify_datasets.py --config configs/train.yaml
# 或
make dataset-verify
```

也可通过 `--sources` 覆盖要核验的数据来源（逗号分隔的 glob 列表，默认取 `data.train_sources` + `data.val_sources`）。

### 输入

- `configs/train.yaml` 中 `data.train_sources` / `data.val_sources` 指定的数据来源目录（位于服务器端）。
- 每个数据来源目录包含 `images/*.png` 和一个 CVAT XML 标注文件（或仅有 `images/*.png` 的纯负样本目录；负样本目录的标准形式为 `<来源>/images/*.png`，parser 也兼容 PNG 直接平铺在来源目录根部的旧布局）。

### 操作

对每个解析出的数据来源目录：

1. 检查目录结构（`images/` + XML，或仅有 `images/` 的负样本目录）。
2. 解析 XML（优先 `cvat_reviewed.xml`，其次 `cvat_autolabel.xml`），统计标签分布（Left/Right/no_hand/ignore_for_training/unknown_handedness）。
3. 交叉核对：XML 中引用的每张图像必须真实存在于 `images/`；磁盘上的每张图像应被 XML 引用（孤儿图像仅告警）。
4. 逐张加载所有 PNG（PIL），确认无损坏文件、尺寸均为 256x256（非灰度模式仅告警）。
5. 无 `images/` 且无 XML 的目录（如 `old/`、`NegativeTrain/` 容器目录）自动跳过——与 `hand_classifier/parser.py` 的收集逻辑一致。

### 输出

- 终端打印每个来源的核验报告（状态、图像数、XML 图像数、标签分布、图像模式、错误/告警列表）与汇总。
- 退出码：0 = 全部正常；2 = 存在错误（XML 无法解析、XML 引用图像缺失、图像损坏、尺寸不符等）。

### 原理

在训练前验证数据仓库可用性，避免训练/评估时因缺图或坏图导致静默跳过（parser 对缺失图像是静默跳过的）。此阶段与 `dataset_stats` 的差别：`dataset_stats` 只统计标签分布，`verify_datasets` 做结构与图像完整性核验。

---

## 阶段 2：数据集统计

### 命令

```bash
python scripts/dataset_stats.py --config configs/train.yaml
```

### 输入

- `configs/train.yaml` 中 `data.train_sources` 指定的数据来源目录（位于服务器端）。
- 每个数据来源目录包含 `images/*.png` 和一个 CVAT XML 标注文件。

### 操作

- 遍历 `data.train_sources` 中列出的所有数据来源目录。
- 对每个数据来源，解析 CVAT XML 文件（优先选择 `cvat_reviewed.xml`，其次 `cvat_autolabel.xml`）。
- 统计每个数据来源中 Left、Right 样本数，以及被排除的样本（ignore_for_training、no_hand、unknown_handedness）。

### 输出

- 在终端打印每个数据来源的标签分布表（Left/Right 数量及比例）。

### 原理

在训练前验证数据完整性，了解类别分布情况。数据集加入新的 rain/thick 手势来源（`eos_2.1-*`）后，全局 Left:Right 比例趋近 1:1，但训练时仍使用平衡类别权重以应对个别来源的偏斜。

---

## 阶段 3：训练

### 命令

```bash
python scripts/train.py --config configs/train.yaml
```

也可用仓库根目录的 `run.sh` 一键执行 核验→统计→训练→评估→导出 全流程（训练完成后自动把 best checkpoint 传给评估与导出步骤）。

### 输入

- `configs/train.yaml` - 包含所有训练超参数与模型选择（`model.version` + `model.architecture`）。
- `autodl-tmp/DatasetFab/HCFTrainSource/*` 下的数据来源目录（含 10 个新增 `eos_2.1-*` rain/thick 训练来源）。

### 操作

1. **数据收集**：解析所有数据来源目录，收集带有 Left/Right/no_hand 标签的样本。
2. **数据集划分**：当前配置指定验证集为 `HCFEvalSource/*`（含 2 个新增 rain/thick 验证来源：`eos_2.1-white-mid-bright-rainleft-val-s06-soar`、`eos_2.1-white-mid-bright-thickright-val-s06-dragon`，通过 glob 自动发现），其余数据来源全部用作训练集。程序自动从训练集中排除验证集来源，避免数据泄漏。若 `val_sources` 为空，则回退到在每个数据来源内部按标签进行 90/10 分层抽样的划分方式。当前不划分测试集。
3. **模型构建**：按 `model.version` + `model.architecture` 构建模型：
   - v1.0：MobileNetV3-Small/Large，第一层卷积的 RGB 三通道预训练权值平均为单通道，分类头替换为双分类头。
   - v2.0 自定义系列：标准卷积 / 多分支卷积 / CNN+Transformer 混合模型，从零训练。
   - v2.0 迁移系列：ImageNet 预训练主干（ResNet50/ConvNeXt/EfficientNetV2/ViT），首层卷积 RGB 权值平均为单通道，位置编码（ViT）插值到 256px 输入。
4. **训练循环**：
   - 损失函数：`CrossEntropyLoss`，使用平衡类别权重（逆频率）。
   - **批量采样比例控制**（config `sampling`）：当配置存在 `sampling` 节时，训练 DataLoader 使用 `StratifiedBatchSampler`，每个 batch 严格按目标比例组成：`round(batch_size * no_hand_ratio)` 个 no_hand 样本 + 其余 has_hand 样本按 `left_right_ratio` 拆分为 Left/Right（handedness 未知的有手样本作为兜底）。负样本池远大于正样本池，此机制保证 no_hand 不会淹没每个 batch，hand presence 分类头不会因训练数据失衡而偏向某一类。
   - 优化器：`AdamW`，采用差异化学习率（backbone 使用基础 lr，分类头使用 lr×10）。
   - 学习率调度：`CosineAnnealingLR` + 线性预热（5 个 epoch）。
   - 启用自动混合精度（AMP，仅 GPU）。
   - 数据增强：水平翻转（同步交换 0↔1 标签）、随机仿射变换、色彩抖动、随机擦除。
5. **早停机制**：监控验证损失（val_loss），patience=15 个 epoch。

### 输出

产物按模型系列/架构分目录存放（下文 `<model_dir>` 表示 `paths.output_root/<model.version>/<model.architecture>/`，未配置 `output_root` 时退化为旧结构 `outputs/`）：

- `<model_dir>/splits.json` - 训练集和验证集的划分元数据。
- `<model_dir>/checkpoints/best.pth` - 最佳模型检查点（按 val_loss 最低）。
- `<model_dir>/checkpoints/last.pth` - 最新模型检查点。
- `<model_dir>/train/metrics.jsonl` - 每个 epoch 的训练指标。

### 参数调整原理

- `model.version` / `model.architecture`：选择训练哪个系列的哪个模型；v2.0 系列参数远大于 v1.0，精度优先、延迟不敏感。两者同时决定产物目录（见 `paths.output_root`），因此**不同模型互不覆盖**，同一模型重复训练仍会覆盖自身旧结果。
- `paths.output_root`：产物根目录（默认 `../autodl-tmp/TrainFab/outputs`）。train/evaluate/export_onnx/cvat_label_test 四个入口都会据此自动定位输入与输出路径（`checkpoint_dir`、`metrics_dir`、`splits_dir`、`eval_dir`、`onnx_path`），无需手工传参；缺省该键时使用配置中的显式旧式路径，向后兼容。
- `training.batch_size: 64`：对于 256×256 单通道图像，RTX 3090 24GB 显存可轻松容纳（v2 大模型建议保持 64，必要时降低到 32）。
- `training.learning_rate: 0.0001`：微调预训练模型的标准学习率；v2.0 自定义系列（从零训练）建议提高到 `0.0003`~`0.001`。
- `training.head_lr_multiplier: 10`：新分类头未经过预训练，需要比 backbone 更快的收敛速度。
- `training.warmup_epochs: 5`：预热有助于在差异化学习率配置下稳定早期训练。
- `training.class_weights: "balanced"`：补偿各来源 Left:Right 类别不平衡。**当 `sampling` 节存在时**，类权重改为由目标采样比例推导（有手/无手按 `no_hand_ratio` 的逆频率，左右手按 `left_right_ratio` 的逆频率），避免与采样机制双重补偿。
- `training.handedness_loss_weight: 1.2` 与 `hand_presence.loss_weight: 1.0`：多任务损失权重，总损失 = `1.2 × loss_handedness + 1.0 × loss_hand_presence`。handedness 权重略高，使模型优先保证左右手判断精度（hand presence 精度仍由类权重与采样比例保障）。权重可调：提高 handedness 权重（如 1.5）进一步偏置，降低则更均衡。早停/最佳检查点选择的 val_loss 同样采用该加权组合，因此最佳模型选择也偏向 handedness 精度。
- `sampling.no_hand_ratio: 0.3`：每个 batch 中 no_hand 样本的目标占比。当前训练集原始比例约为 has_hand:no_hand = 46:54（负样本池更大），若不控制，多数 batch 会被负样本主导，presence 头将偏向把一切判为 no_hand；取 0.3 使每个 batch 保持 70:30 的有手/无手比例，负样本仍然充足但不再主导。比例可调：调大则 no_hand 召回优先，调小则 no_hand 误报优先。
- `sampling.left_right_ratio: [0.5, 0.5]`：有手样本中 Left/Right 的目标占比，取 50:50 使 handedness 头在平衡分布上训练（配合 50:50 采样，handedness 类权重自动为均匀权重）。
- `sampling` 机制只作用于**训练**；验证集按真实分布整体使用（不重采样），保证验证指标反映真实数据。
- `augmentation.horizontal_flip_prob: 0.5`：水平翻转后必须同步交换 0↔1 标签，保证标签正确性。
- `augmentation.rotation_degrees: 10`：小角度旋转可模拟手部姿态的自然变化。

---

## 阶段 4：评估

### 命令

```bash
python scripts/evaluate.py --config configs/evaluate.yaml [--checkpoint <model_dir>/checkpoints/best.pth]
```

### 输入

- 训练好的模型检查点（默认自动定位到 `<model_dir>/checkpoints/best.pth`）。
- **配置自动对齐**：检查点内保存了训练时的配置；若 `evaluate.yaml` 的 model 段与检查点不一致，程序打印警告并自动采用检查点的模型配置（模型构建与输出目录均跟随），因此无需手工同步三个配置文件的 model 段。
- 验证集（由划分产生或来自 `data.val_sources`）。

### 操作

1. 加载最佳检查点。
2. 对验证集进行推理（不做数据增强，仅做归一化）。
3. 计算指标：准确率（Accuracy）、各类别精确率/召回率/F1值、混淆矩阵、ROC-AUC。
4. 计算每个数据来源的单独准确率。

### 输出

- 终端输出：所有评估指标。
- `<model_dir>/eval/val_metrics.json` - 完整评估指标 JSON 文件（`--output-dir` 可覆盖）。
- 若指定了测试来源，还会输出 `<model_dir>/eval/test_metrics.json`。

---

## 阶段 5：ONNX 导出

### 命令

```bash
python scripts/export_onnx.py --config configs/export_onnx.yaml [--checkpoint <model_dir>/checkpoints/best.pth] [-o <model_dir>/model.onnx]
```

### 输入

- 训练好的模型检查点（默认自动定位到 `<model_dir>/checkpoints/best.pth`；**配置自动对齐**机制同阶段 4，`export_onnx.yaml` 的 model 段无需与训练配置同步）。

### 操作

1. 加载模型和检查点。
2. 将模型移至 CPU。
3. 使用 `torch.onnx.export` 导出（opset=13，动态 batch，开启常量折叠）。
4. 验证：使用 `onnx.checker` 校验 ONNX 模型结构，使用 `onnxruntime` 进行推理并检查输出形状。

### 输出

- `<model_dir>/model.onnx` - 支持动态 batch 的 ONNX 模型文件（默认输出路径，`-o` 可覆盖）。

### 原理

- opset=13 是广泛支持的 ONNX 算子集版本，覆盖 v1.0/v2.0 所有模型算子（GELU、多分支卷积、手动实现的注意力等均为 opset-13 兼容算子）。
- 动态 batch 允许推理阶段处理任意批量大小。
- 验证步骤确保导出后的 ONNX 模型在部署前是有效的。
- 无论 v1.0 还是 v2.0，导出的 ONNX 输入/输出接口完全一致（`input` / `handedness` / `hand_presence`），下游标注流程无需任何改动。

---

## 阶段 6：CVAT 标签导出测试

### 命令

```bash
python scripts/cvat_label_test.py --config configs/cvat_label_test.yaml [--checkpoint <model_dir>/model.onnx] [--xml <path>]
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
