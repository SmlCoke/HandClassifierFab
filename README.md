# 手部左/右手二分类器 (HCF)

为 Iris 系列模型的数据集自动标注流程设计的轻量级手部左右手（Left/Right）二分类模型。

模型接收 256x256 灰度（单通道）Hand ROI 图像，输出二分类结果：`Left`（0）或 `Right`（1），与 Iris 系列模型的标签约定保持一致。

## 目录结构

### 本仓库

```
HandClassifierFab/
├── configs/
│   ├── train.yaml                 # 训练配置（数据、模型版本、超参数、增强）
│   ├── evaluate.yaml              # 评估配置
│   ├── export_onnx.yaml           # ONNX 导出配置
│   ├── cvat_label_test.yaml       # CVAT 标签导出测试配置
│   └── infer.yaml                 # 推理配置（负样本筛选）
├── data/
│   ├── examples/
│   │   └── dataset1/              # 数据集样例（供理解格式使用）
│   │       ├── images/*.png       # 32 张 256x256 灰度 Hand ROI 图像
│   │       └── cvat_reviewed.xml  # 人工复核的金标准标注
│   └── dataset_test/
│       └── complex-near-bright-random-val-s01-peak/
│           ├── images/*.png       # 485 张灰度 Hand ROI 图像
│           └── cvat_autolabel.xml # Iris 自动标注系统生成的标签
├── docs/
│   ├── HCF_system/
│   │   ├── HCF_annotating_workflow.md  # 工作流文档（命令 + 原理）
│   │   ├── HCF_quick_start.md          # 快速上手文档（仅命令）
│   │   └── HCF_qa.md                   # 问答记录文档
│   └── prompts/
│       ├── prompt-1.md                 # 原始需求说明
│       ├── prompt-2.md                 # 双头模型需求说明
│       ├── prompt-3.md                 # v2.0 模型需求说明
│       └── tmp.txt                     # AutoDL 服务器环境信息
├── hand_classifier/
│   ├── __init__.py              # 包入口，导出所有公共接口
│   ├── config.py                # YAML 配置加载
│   ├── parser.py                # CVAT XML 解析器
│   ├── dataset.py               # PyTorch Dataset、数据增强、数据集划分
│   ├── trainer.py               # 训练循环（AMP、预热、早停）
│   ├── evaluator.py             # 模型评估（准确率、F1、混淆矩阵）
│   ├── exporter.py              # ONNX 导出及验证
│   ├── relabel.py               # CVAT XML 标签替换（字节保留正则）
│   └── infer.py                 # ONNX 批量推理（负样本筛选）
├── models/
│   ├── __init__.py              # 模型包入口
│   ├── factory.py               # 模型构建器注册表（v1/v2 版本感知）
│   ├── v1/                      # v1.0 系列：原始 MobileNetV3 模型
│   │   ├── __init__.py
│   │   └── mobilenetv3.py       # 单通道 MobileNetV3-Small/Large（双头）
│   └── v2/                      # v2.0 系列：高精度大参数量模型（双头）
│       ├── __init__.py
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
├── tests/
│   ├── conftest.py              # 共享测试 fixtures
│   ├── test_cvat_parser.py      # CVAT XML 解析器测试
│   ├── test_model.py            # v1.0 模型架构测试
│   ├── test_v2_model.py         # v2.0 模型架构测试（含 ONNX 导出契约）
│   ├── test_dataset.py          # Dataset 及数据增强测试
│   ├── test_split.py            # 数据划分策略测试
│   ├── test_preprocessing.py    # 预处理 / 归一化测试
│   ├── test_cvat_label_export.py # CVAT XML 标签替换测试
│   └── test_onnx_export.py      # ONNX 导出验证测试
├── outputs/                     # 运行时生成（gitignore；服务器端实际输出见 paths.output_root）
│   └── <version>/<architecture>/  # 按模型系列/架构分目录（见"输出目录"一节）
├── .gitignore
├── CLAUDE.md                    # 仓库级 Agent 行为规范
├── Makefile                     # make 快捷命令
├── README.md                    # 本文件
└── requirements.txt             # Python 依赖
```

### AutoDL 服务器数据集目录

```
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

## 模型架构

### v1.0 系列（原始版本，`models/v1/`）

- `mobilenet_v3_small`：MobileNetV3-Small，第一层卷积由 RGB 三通道改造为单通道灰度输入（对 ImageNet 预训练 RGB 权值取平均），分类头替换为双分类头（handedness + hand_presence）。参数约 150 万。
- `mobilenet_v3_large`：MobileNetV3-Large 同构改造，参数约 420 万。

### v2.0 系列（高精度大参数量版本，`models/v2/`）

为提升 handedness / hand_presence 精度而设计，牺牲延迟换取精度，**输入输出格式与 v1.0 完全一致**（输入 `(N,1,256,256)` 灰度张量，输出 dict：`handedness` 与 `hand_presence` 各 `(N,2)` logits），因此下游调用方式不变。

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
- 所有算子兼容 ONNX opset-13；`scripts/export_onnx.py` 导出的 ONNX 输入/输出名与 v1.0 相同（`input` / `handedness` / `hand_presence`，动态 batch）。

### 模型版本选择

`configs/train.yaml`、`configs/evaluate.yaml`、`configs/export_onnx.yaml` 中通过 `model.version`（`v1` / `v2`）与 `model.architecture` 选择每次训练/评估/导出所用的模型，例如：

```yaml
model:
  version: "v2"
  architecture: "v2_convnet_l"
```

## 数据集划分策略

- **训练集**（HCFTrainSource 全部来源，含 10 个新增 rain/thick 来源 + NegativeTrain 负样本，约 15000 张）：除验证集来源外的全部数据来源
- **验证集**（HCFEvalSource 全部来源，含 2 个新增 rain/thick 验证来源 + NegativeVal 负样本）：`val_sources` glob 自动发现
- **测试集**：暂不划分。代码中保留了 Test DataLoader 和划分接口（通过配置中的 `test_sources` 和 `test_ratio` 控制），后续用户自行标注测试数据集后可直接使用
- 划分信息保存至 `outputs/splits.json`

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 查看数据集统计信息
python scripts/dataset_stats.py --config configs/train.yaml

# 核验数据集完整性（XML/图像/标签/尺寸）
python scripts/verify_datasets.py --config configs/train.yaml

# 训练
python scripts/train.py --config configs/train.yaml

# 评估
python scripts/evaluate.py --config configs/evaluate.yaml

# 导出 ONNX
python scripts/export_onnx.py --config configs/export_onnx.yaml

# CVAT 标签导出测试
python scripts/cvat_label_test.py --config configs/cvat_label_test.yaml

# 推理：筛选无手负样本
python scripts/infer.py --config configs/infer.yaml

# 运行测试
python -m pytest tests/ -v
```

也可以使用 Makefile：

```bash
make setup           # 安装依赖
make dataset-stats   # 查看数据来源标签分布
make dataset-verify  # 核验数据集完整性
make train           # 训练模型
make evaluate        # 评估模型
make export-onnx     # 导出 ONNX 模型
make cvat-label-test # 测试 CVAT 自动标注标签替换
make infer           # 推理：筛选低 hand_presence 负样本
make test            # 运行全部单元测试
make all             # test + train + evaluate + export-onnx + cvat-label-test
make clean           # 清理 outputs/、__pycache__/、.pytest_cache/
```

## 配置说明

配置文件按功能拆分，位于 `configs/` 目录下：

| 文件 | 用途 |
|------|------|
| `train.yaml` | 数据来源、模型版本与架构、训练超参数、批量采样比例、数据增强 |
| `evaluate.yaml` | 评估数据来源、模型版本与架构 |
| `export_onnx.yaml` | ONNX 导出参数、模型版本与架构 |
| `cvat_label_test.yaml` | CVAT 标签导出测试配置 |
| `infer.yaml` | 推理配置（ONNX 模型、输入/输出目录、阈值） |

### 批量采样比例（`sampling`）

当前负样本（no_hand）池远大于正样本（has_hand）池，若不做控制，每个 batch 会被负样本主导，导致 hand presence 分类头偏向某一类。`configs/train.yaml` 的 `sampling` 节启用按 batch 精确配比的采样（仅作用于训练，验证集按真实分布整体使用）：

```yaml
sampling:
  enabled: true
  no_hand_ratio: 0.3            # 每个 batch 中 no_hand 样本的目标占比（0~1）
  left_right_ratio: [0.5, 0.5]  # 有手样本中 Left / Right 的目标占比（和为 1）
```

启用后，训练集类权重改为由这些目标比例推导（而非原始数据计数），避免双重补偿。每个 epoch 为一轮完整的有手样本遍历，负样本每 epoch 随机取子集（等价于随机负样本挖掘）。

### 多任务损失权重

总损失 = `training.handedness_loss_weight × loss_handedness + hand_presence.loss_weight × loss_hand_presence`。默认 `handedness_loss_weight: 1.2`、`hand_presence.loss_weight: 1.0`，handedness（左右手）损失占比略高，优先保证 handedness 精度；早停/最佳检查点选择的 val_loss 同样采用该加权组合。权重可在 `configs/train.yaml` 中调整（例如提高到 1.5 进一步偏置 handedness）。

### 输出目录（`paths.output_root`）

训练、评估、ONNX 导出产物按**模型系列 + 具体架构**分目录存放，不同模型互不覆盖：

```
<paths.output_root>/<model.version>/<model.architecture>/
├── checkpoints/best.pth, last.pth   # 训练检查点
├── train/metrics.jsonl              # 训练指标
├── eval/val_metrics.json            # 评估指标
├── splits.json                      # 数据集划分信息
└── model.onnx                       # 导出的 ONNX 模型
```

例如默认配置（v1 / mobilenet_v3_small）产出在 `../autodl-tmp/TrainFab/outputs/v1/mobilenet_v3_small/`。所有入口脚本（train / evaluate / export_onnx / cvat_label_test）都会根据 `paths.output_root` + `model.version` + `model.architecture` 自动定位输入检查点和输出目录，无需手工传路径；未配置 `output_root` 时保持旧的显式路径行为（向后兼容）。

**评估/导出自动跟随训练配置**：训练检查点内保存了训练时的完整配置。评估/导出时若 checkpoint 的训练配置与当前配置文件（`evaluate.yaml` / `export_onnx.yaml`）不一致（例如配置文件仍指向别的架构），程序会打印警告并自动采用 checkpoint 的模型配置构建模型、把产物输出到该模型自己的目录——因此这三个配置文件的 `model` 段无需手工保持同步。仓库根目录的 `run.sh` 已按此流程将训练得到的 checkpoint 显式传给 evaluate/export，训练什么就评估/导出什么。

## 数据来源格式

每个数据来源目录的标准格式：

```
├── images/*.png          # 256x256 灰度 Hand ROI 图像（单通道）
└── cvat_reviewed.xml     # CVAT 1.1 格式的人工复核金标准标注
```

自动标注版本：

```
├── images/*.png
└── cvat_autolabel.xml    # Iris 自动标注系统生成的标签（handedness 为 unknown_handedness）
```

纯负样本版本（无 XML，全部图像视为 no_hand）：

```
└── images/*.png
```

负样本目录的标准形式为 `<来源>/images/*.png`；parser 也兼容 PNG 直接平铺在来源目录根部的旧布局（`verify_datasets` 会给出告警提示规范化）。

## 服务器环境

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

```bash
# 服务器端安装（CUDA 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 文档

- [工作流文档](docs/HCF_system/HCF_annotating_workflow.md) - 每个操作步骤的命令、操作内容和参数调整原理
- [快速上手文档](docs/HCF_system/HCF_quick_start.md) - 仅含命令的简化版本
