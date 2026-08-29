<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign minimal logo" width="116" />

<h1>HandClassifierFab（HCF）</h1>

**AetherSign 数据集自动标注链路 · 手部左/右手双头分类器训练与导出系统**

[![Archive](https://img.shields.io/badge/Status-Competition_Final-8B5CF6?style=flat-square)](#-i-项目定位与归档状态) [![Tag](https://img.shields.io/badge/Tag-HCF_1.0_final-0891B2?style=flat-square)](https://github.com/SmlCoke/HandClassifierFab/tree/HCF-1.0-final) [![Tests](https://img.shields.io/badge/Tests-86_Passed-059669?style=flat-square)](#-ix-依赖管理) [![ROI](https://img.shields.io/badge/Hand_ROI-256%C3%97256-2563EB?style=flat-square)](#-vii-数据契约)

[项目定位](#-i-项目定位与归档状态) · [系统与模型](#-ii-系统与模型) · [文档入口](#-iii-文档入口) · [快速开始](#-iv-快速开始) · [目录结构](#-v-目录结构) · [配置说明](#-vi-配置说明) · [数据契约](#-vii-数据契约) · [复现环境](#-viii-复现环境) · [依赖管理](#-ix-依赖管理)

</div>

---

## ✦ I. 项目定位与归档状态

### 1.1 项目定位

HandClassifierFab（HCF）是 AetherSign 手语识别系统中 **Iris（Hand Landmarker）系列模型数据集自动化标注链路**的配套训练仓库。RTMPose-m Hand5 等教师模型可以从 Hand ROI 图像中检测手部关键点，但**无法判断左右手**；HCF 补上这一环：接收 `256×256` 单通道灰度 Hand ROI 图像，输出 handedness（Left/Right）与 hand_presence（no_hand/has_hand）双头分类结果，与 Iris 模型的标签约定保持一致。

#### 1.1.1 双头输出契约

模型输出为 dict：`{"handedness": Tensor(N,2), "hand_presence": Tensor(N,2)}`，两个头共享 backbone。训练好的模型导出为 ONNX（两个输出头，动态 batch），用于对 CVAT 自动标注 XML 重新标注，也可用于批量筛选低 hand_presence 的负样本。

#### 1.1.2 模型系列

| 系列 | 定位 | 参数规模 |
| :-- | :-- | :-- |
| v1.0（`models/v1/`） | 原始 MobileNetV3-Small/Large 双头模型，轻量 | 约 150 万 / 420 万 |
| v2.0（`models/v2/`） | 为提升精度而设计的大参数量模型，牺牲延迟换取精度，**不参与板端部署**，仅用于数据自动化标注 | 约 10M ~ 85M |

### 1.2 比赛归档状态

> [!IMPORTANT]
> AetherSign 已于 2026-08-25 完成全国总决赛答辩并获得全国一等奖。本仓库针对本届比赛的使命已经完成，最终可复现代码状态由 annotated tag **`HCF-1.0-final`** 固定。

#### 1.2.1 Git 归档范围

tag 保存代码、配置、测试与文档。模型权重、ONNX 产物（`assets/`，按既有策略不纳入 Git）与服务器端数据仓（`autodl-tmp/DatasetFab/`）独立保存，不包含在 Git tag 中。

#### 1.2.2 正式使用模型

- **v1.0 系列**：`mobilenet_v3_large` 已接入 HLMF（HandLandmarkerFab）自动标注链路，作为 runtime ROI 分类的权威来源（hand presence + handedness）。
- **v2.0 系列**：精度优先的数据自动化标注模型系列，全部 8 个架构保留完整训练/评估/导出链路。

### 1.3 项目背景文档

本届比赛的完整项目背景与系统架构说明见 [project-12.md](./project-12.md)。

---

## ◇ II. 系统与模型

### 2.1 双头模型结构

共享 backbone + 双分类头（`handedness_head` / `hand_presence_head`）。v1.0 首层卷积由 RGB 三通道改造为单通道灰度输入（ImageNet 预训练 RGB 权值取平均）；v2.0 迁移系列（ResNet50/ConvNeXt/EfficientNetV2/ViT）同样做单通道适配，ViT 位置编码插值到 256px 输入。

### 2.2 v1.0 系列

- `mobilenet_v3_small`：MobileNetV3-Small 同构改造，参数约 150 万。
- `mobilenet_v3_large`：MobileNetV3-Large 同构改造，参数约 420 万。

### 2.3 v2.0 系列

输入输出格式与 v1.0 完全一致（输入 `(N,1,256,256)` 灰度张量，输出 dict：`handedness` 与 `hand_presence` 各 `(N,2)` logits），下游调用方式不变。

| 架构名 | 类型 | 参数规模 |
|--------|------|----------|
| `v2_convnet_s` | 标准 3x3 卷积 + SE 注意力（ResNet 风格） | ~13M |
| `v2_convnet_l` | 标准卷积加深加宽 + SE 注意力 | ~64M |
| `v2_multibranch` | Inception 风格多分支卷积 + SE 注意力 | ~10M |
| `v2_hybrid_s` | 标准卷积 stem + Transformer 编码器（预归一化 MHA） | ~16M |
| `v2_hybrid_l` | 同上加深加宽 | ~41M |
| `v2_resnet50` | ImageNet 预训练 ResNet-50 主干（1 通道适配） | ~24M |
| `v2_convnext_tiny` | ImageNet 预训练 ConvNeXt-Tiny 主干（1 通道适配） | ~28M |
| `v2_efficientnet_v2_s` | ImageNet 预训练 EfficientNetV2-S 主干（1 通道适配） | ~20M |
| `v2_vit_b16` | ImageNet 预训练 ViT-B/16（位置编码插值到 256px 输入） | ~85M |

说明：

- 自定义 CNN / 混合模型（`v2_convnet_*`、`v2_multibranch`、`v2_hybrid_*`）从零训练，无预训练权重。
- 迁移模型（`v2_resnet50` 等）通过 `model.pretrained: true` 加载 ImageNet 预训练权值，首层卷积的 RGB 权值取平均适配单通道输入。
- 所有算子兼容 ONNX opset-13；导出的 ONNX 输入/输出名与 v1.0 相同（`input` / `handedness` / `hand_presence`，动态 batch）。

### 2.4 模型版本选择

`configs/train.yaml`、`configs/evaluate.yaml`、`configs/export_onnx.yaml` 中通过 `model.version`（`v1` / `v2`）与 `model.architecture` 选择每次训练/评估/导出所用的模型，例如：

```yaml
model:
  version: "v2"
  architecture: "v2_convnet_l"
```

---

## 🧭 III. 文档入口

### 3.1 核心入口文档

| 文档 | 作用 |
| :-- | :-- |
| [完整工作流](docs/HCF_system/HCF_annotating_workflow.md) | 六个阶段（核验 → 统计 → 训练 → 评估 → ONNX 导出 → CVAT 标签导出测试）的命令、操作内容与配置原理 |
| [快速上手](docs/HCF_system/HCF_quick_start.md) | 端到端操作命令速查（仅命令） |
| [常见问题与解答](docs/HCF_system/HCF_qa.md) | 已记录的仓库问题与答案 |

### 3.2 项目背景

[project-12.md](./project-12.md)：本届比赛的项目背景与系统架构全景。

---

## 🚀 IV. 快速开始

### 4.1 环境准备

```bash
# 创建 conda 环境（Python 3.10）
conda create -n hand_classifier python=3.10
conda activate hand_classifier

# 安装 PyTorch（CUDA 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt
```

### 4.2 一键流水线

仓库根目录的 `run.sh` 按 核验 → 统计 → 训练 → 评估 → 导出 顺序执行全流程（训练完成后自动把 best checkpoint 传给评估与导出步骤）：

```bash
./run.sh
```

### 4.3 分步命令

```bash
# 1. 核验数据集完整性（XML/图像/标签/尺寸）
python scripts/verify_datasets.py --config configs/train.yaml

# 2. 查看数据来源标签分布
python scripts/dataset_stats.py --config configs/train.yaml

# 3. 训练
python scripts/train.py --config configs/train.yaml

# 4. 评估
python scripts/evaluate.py --config configs/evaluate.yaml

# 5. 导出 ONNX
python scripts/export_onnx.py --config configs/export_onnx.yaml

# 6. CVAT 标签导出测试
python scripts/cvat_label_test.py --config configs/cvat_label_test.yaml

# 7. 推理：筛选低 hand_presence 负样本
python scripts/infer.py --config configs/infer.yaml

# 8. 运行测试
python -m pytest tests/ -v
```

### 4.4 Makefile 快捷命令

```bash
make setup             # pip install -r requirements.txt
make dataset-verify    # 核验数据集完整性
make dataset-stats     # 打印标签分布
make train             # 训练模型
make evaluate          # 运行评估
make export-onnx       # 导出 ONNX 模型
make cvat-label-test   # 运行 CVAT 自动标注标签替换
make infer             # 推理：筛选低 hand_presence 负样本
make test              # 运行全部单元测试
make all               # test + train + evaluate + export-onnx + cvat-label-test
make clean             # 清理 outputs/、__pycache__/、.pytest_cache/
```

---

## 🗂 V. 目录结构

### 5.1 本仓库

```text
HandClassifierFab/
├── configs/
│   ├── train.yaml                 # 训练配置（数据、模型版本、超参数、增强、采样）
│   ├── evaluate.yaml              # 评估配置
│   ├── export_onnx.yaml           # ONNX 导出配置
│   ├── cvat_label_test.yaml       # CVAT 标签导出测试配置
│   └── infer.yaml                 # 推理配置（负样本筛选）
├── docs/
│   ├── assets/                    # 视觉素材（AetherSign logo，单独维护）
│   ├── HCF_system/
│   │   ├── HCF_annotating_workflow.md  # 工作流文档（命令 + 原理）
│   │   ├── HCF_quick_start.md          # 快速上手文档（仅命令）
│   │   └── HCF_qa.md                   # 问答记录文档
│   └── prompts/
│       └── prompt-4.md            # 最终归档说明（其余 prompts 不纳入版本库）
├── hand_classifier/
│   ├── __init__.py              # 包入口，导出所有公共接口
│   ├── config.py                # YAML 配置加载、输出路径推导、checkpoint 配置对齐
│   ├── parser.py                # CVAT XML 解析器
│   ├── dataset.py               # PyTorch Dataset、数据增强、数据集划分、批量采样
│   ├── trainer.py               # 训练循环（AMP、预热、早停）
│   ├── evaluator.py             # 模型评估（准确率、F1、混淆矩阵）
│   ├── exporter.py              # ONNX 导出及验证
│   ├── relabel.py               # CVAT XML 标签替换（字节保留正则）
│   └── infer.py                 # ONNX 批量推理（负样本筛选）
├── models/
│   ├── __init__.py              # 模型包入口
│   ├── factory.py               # 模型构建器注册表（v1/v2 版本感知）
│   ├── v1/                      # v1.0 系列：原始 MobileNetV3 模型
│   │   └── mobilenetv3.py       # 单通道 MobileNetV3-Small/Large（双头）
│   └── v2/                      # v2.0 系列：高精度大参数量模型（双头）
│       ├── blocks.py            # 共享模块（SE 注意力、DualHead 等）
│       ├── convnet.py           # 标准卷积 / 多分支卷积 CNN
│       ├── hybrid.py            # CNN + Transformer 混合模型
│       └── transfer.py          # ImageNet 预训练主干（ResNet50/ConvNeXt 等）
├── scripts/
│   ├── train.py                 # 训练入口脚本
│   ├── evaluate.py              # 评估入口脚本
│   ├── export_onnx.py           # ONNX 导出入口脚本
│   ├── cvat_label_test.py       # CVAT 标签导出测试入口脚本
│   ├── dataset_stats.py         # 数据集统计信息入口脚本
│   └── verify_datasets.py       # 数据集完整性核验入口脚本
├── tests/                       # 单元测试（86 项，含 v2 模型与 ONNX 导出契约）
├── data/                        # 本地样例数据（不纳入版本库）
├── assets/                      # 模型产物（不纳入版本库，服务器端独立保存）
├── .gitignore
├── CLAUDE.md                    # 仓库级 Agent 行为规范
├── Makefile                     # make 快捷命令
├── README.md                    # 本文件
├── project-12.md                # 项目背景与系统架构全景文档
├── requirements.txt             # Python 依赖
└── run.sh                       # 一键流水线脚本（核验→统计→训练→评估→导出）
```

### 5.2 AutoDL 服务器数据集目录

```text
autodl-tmp/
└── DatasetFab/
    ├── HCFTrainSource/                # 训练数据集（含 10 个新增 rain/thick 来源）
    │    ├── complex-far-bright-random-test-s01-peak/   # [训练]
    │    │   ├── images/*.png          # 约 600 张 256x256 灰度 Hand ROI 图像
    │    │   └── cvat_reviewed.xml     # 人工复核的金标准标注
    │    ├── ...                       # 其余训练来源（glob 自动发现）
    │    ├── eos_2.1-white-mid-bright-rainleft-val-s06-dragon/  # [训练] 新增
    │    ├── ...                       # 其余 9 个 eos_2.1 rain/thick 训练来源
    │    ├── NegativeTrain/            # 纯负样本（无 XML，全部视为 no_hand）
    │    └── old/                      # 旧版本数据集（自动忽略，不参与训练）
    ├── HCFEvalSource/                 # 验证数据集（glob 自动发现）
    │    ├── eos_2.1-white-mid-bright-rainleft-val-s06-soar/   # [验证] 新增
    │    ├── eos_2.1-white-mid-bright-thickright-val-s06-dragon/ # [验证] 新增
    │    ├── eos_2.0-*/                # 既有验证来源
    │    ├── NegativeVal/              # 纯负样本验证来源
    │    └── old/                      # 旧版本数据集（自动忽略）
    ├── HCFCVATTestSource/             # CVAT 标签导出测试数据集
    └── NegativeSamples/               # 历史负样本（未用于当前训练）
```

---

## ⚙️ VI. 配置说明

### 6.1 配置文件

配置文件按功能拆分，位于 `configs/` 目录下：

| 文件 | 用途 |
|------|------|
| `train.yaml` | 数据来源、模型版本与架构、训练超参数、批量采样比例、数据增强 |
| `evaluate.yaml` | 评估数据来源、模型版本与架构 |
| `export_onnx.yaml` | ONNX 导出参数、模型版本与架构 |
| `cvat_label_test.yaml` | CVAT 标签导出测试配置 |
| `infer.yaml` | 推理配置（ONNX 模型、输入/输出目录、阈值） |

### 6.2 批量采样比例（`sampling`）

当前负样本（no_hand）池远大于正样本（has_hand）池，若不做控制，每个 batch 会被负样本主导，导致 hand presence 分类头偏向某一类。`configs/train.yaml` 的 `sampling` 节启用按 batch 精确配比的采样（仅作用于训练，验证集按真实分布整体使用）：

```yaml
sampling:
  enabled: true
  no_hand_ratio: 0.3            # 每个 batch 中 no_hand 样本的目标占比（0~1）
  left_right_ratio: [0.5, 0.5]  # 有手样本中 Left / Right 的目标占比（和为 1）
```

启用后，训练集类权重改为由这些目标比例推导（而非原始数据计数），避免双重补偿。每个 epoch 为一轮完整的有手样本遍历，负样本每 epoch 随机取子集（等价于随机负样本挖掘）。

### 6.3 多任务损失权重

总损失 = `training.handedness_loss_weight × loss_handedness + hand_presence.loss_weight × loss_hand_presence`。默认 `handedness_loss_weight: 1.2`、`hand_presence.loss_weight: 1.0`，handedness（左右手）损失占比略高，优先保证 handedness 精度；早停/最佳检查点选择的 val_loss 同样采用该加权组合。权重可在 `configs/train.yaml` 中调整（例如提高到 1.5 进一步偏置 handedness）。

### 6.4 输出目录（`paths.output_root`）

训练、评估、ONNX 导出产物按**模型系列 + 具体架构**分目录存放，不同模型互不覆盖：

```text
<paths.output_root>/<model.version>/<model.architecture>/
├── checkpoints/best.pth, last.pth   # 训练检查点
├── train/metrics.jsonl              # 训练指标
├── eval/val_metrics.json            # 评估指标
├── splits.json                      # 数据集划分信息
└── model.onnx                       # 导出的 ONNX 模型
```

#### 6.4.1 评估/导出自动跟随训练配置

训练检查点内保存了训练时的完整配置。评估/导出时若 checkpoint 的训练配置与当前配置文件（`evaluate.yaml` / `export_onnx.yaml`）不一致（例如配置文件仍指向别的架构），程序会打印警告并自动采用 checkpoint 的模型配置构建模型、把产物输出到该模型自己的目录——因此这三个配置文件的 `model` 段无需手工保持同步。`run.sh` 已按此流程将训练得到的 checkpoint 显式传给 evaluate/export。

#### 6.4.2 数据集划分

- **训练集**（HCFTrainSource 全部来源，含 10 个新增 rain/thick 来源 + NegativeTrain 负样本）：除验证集来源外的全部数据来源
- **验证集**（HCFEvalSource 全部来源，含 2 个新增 rain/thick 验证来源 + NegativeVal 负样本）：`val_sources` glob 自动发现
- **测试集**：暂不划分。代码中保留了 Test DataLoader 和划分接口（通过配置中的 `test_sources` 和 `test_ratio` 控制），后续可自行标注测试数据集后直接使用
- 划分信息保存至 `outputs/splits.json`

---

## 🔒 VII. 数据契约

### 7.1 标准数据来源格式

每个数据来源目录的标准格式（人工复核金标准）：

```text
├── images/*.png          # 256x256 灰度 Hand ROI 图像（单通道）
└── cvat_reviewed.xml     # CVAT 1.1 格式的人工复核金标准标注
```

### 7.2 自动标注版本

```text
├── images/*.png
└── cvat_autolabel.xml    # Iris 自动标注系统生成的标签（handedness 为 unknown_handedness）
```

### 7.3 纯负样本版本

无 XML 时全部图像视为 no_hand：

```text
└── images/*.png
```

负样本目录的标准形式为 `<来源>/images/*.png`；parser 也兼容 PNG 直接平铺在来源目录根部的旧布局（`verify_datasets` 会给出告警提示规范化）。

---

## ⬡ VIII. 复现环境

### 8.1 服务器实例配置

比赛阶段使用的 AutoDL 服务器基础实例（RTX 3090）：

| 项目 | 配置 |
|------|------|
| GPU | RTX 3090 24GB |
| CPU | 14 vCPU Intel Xeon Gold 6330 @ 2.00GHz |
| 内存 | 90GB |
| 磁盘 | 系统盘 30GB + 数据盘 50GB SSD |
| 操作系统 | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.1.0 |
| CUDA | 12.1 |

### 8.2 安装命令

```bash
# 服务器端安装（CUDA 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🧩 IX. 依赖管理

### 9.1 主环境

主环境由 `requirements.txt` 管理。PyTorch 2.1.0 / CUDA 12.1 下运行时需固定 NumPy `<2` 与 OpenCV `<5`（与 PyTorch 2.1 的 ABI 约束一致），onnxruntime 版本见 requirements。

### 9.2 单元测试

`python -m pytest tests/ -v` 运行全部单元测试（86 项）。无 onnx/onnxruntime 依赖的环境下，ONNX 相关测试自动跳过（`pytest.importorskip`），不影响其余测试收集。

---

<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign" width="52" />

<sub>AetherSign · Eos → Iris → Muse · 全国总决赛一等奖项目归档</sub>

</div>
