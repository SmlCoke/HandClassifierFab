# 手部左/右手二分类器 (HCF)

为 Iris 系列模型的数据集自动标注流程设计的轻量级手部左右手（Left/Right）二分类模型。

模型接收 256x256 灰度（单通道）Hand ROI 图像，输出二分类结果：`Left`（0）或 `Right`（1），与 Iris 系列模型的标签约定保持一致。

## 目录结构

### 本仓库

```
HandClassifierFab/
├── configs/
│   └── hand_classifier.yaml       # 主配置文件（所有超参数）
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
│   │   └── HCF_quick_start.md          # 快速上手文档（仅命令）
│   └── prompts/
│       ├── prompt-1.md                 # 原始需求说明
│       └── tmp.txt                     # AutoDL 服务器环境信息
├── hand_classifier/
│   ├── __init__.py              # 包入口，导出所有公共接口
│   ├── config.py                # YAML 配置加载
│   ├── parser.py                # CVAT XML 解析器
│   ├── dataset.py               # PyTorch Dataset、数据增强、数据集划分
│   ├── trainer.py               # 训练循环（AMP、预热、早停）
│   ├── evaluator.py             # 模型评估（准确率、F1、混淆矩阵）
│   ├── exporter.py              # ONNX 导出及验证
│   └── relabel.py               # CVAT XML 标签替换（字节保留正则）
├── models/
│   ├── __init__.py              # 模型包入口
│   ├── mobilenetv3.py           # 单通道 MobileNetV3-Small/Large
│   └── factory.py               # 模型构建器注册表
├── scripts/
│   ├── train.py                 # 训练入口脚本
│   ├── evaluate.py              # 评估入口脚本
│   ├── export_onnx.py           # ONNX 导出入口脚本
│   ├── cvat_label_test.py       # CVAT 标签导出测试入口脚本
│   └── dataset_stats.py         # 数据集统计信息入口脚本
├── tests/
│   ├── conftest.py              # 共享测试 fixtures
│   ├── test_cvat_parser.py      # CVAT XML 解析器测试
│   ├── test_model.py            # 模型架构测试
│   ├── test_dataset.py          # Dataset 及数据增强测试
│   ├── test_split.py            # 数据划分策略测试
│   ├── test_preprocessing.py    # 预处理 / 归一化测试
│   ├── test_cvat_label_export.py # CVAT XML 标签替换测试
│   └── test_onnx_export.py      # ONNX 导出验证测试
├── outputs/                     # 运行时生成（gitignore）
│   ├── checkpoints/             # 模型检查点
│   ├── train/                   # 训练指标
│   ├── eval/                    # 评估结果
│   └── splits.json              # 数据集划分信息
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
    ├── HCFTrainSource/                # 训练数据集（6 个来源） + 验证数据集（2 个来源）
    │    ├── complex-far-bright-random-test-s01-peak/   # [训练]
    │    │   ├── images/*.png          # 约 600 张 256x256 灰度 Hand ROI 图像
    │    │   └── cvat_reviewed.xml     # 人工复核的金标准标注
    │    ├── complex-far-bright-random-val-s01-peak/     # [训练]
    │    │   ├── images/*.png          # 约 600 张
    │    │   └── cvat_reviewed.xml
    │    ├── complex-mid-bright-random-test-s01-peak/    # [训练]
    │    │   ├── images/*.png          # 约 600 张
    │    │   └── cvat_reviewed.xml
    │    ├── complex-mid-bright-random-val-s01-peak/     # [验证]
    │    │   ├── images/*.png          # 约 600 张
    │    │   └── cvat_reviewed.xml
    │    ├── complex-mid-bright-tandom-val-s01-soar/     # [训练]
    │    │   ├── images/*.png          # 约 1100 张
    │    │   └── cvat_reviewed.xml
    │    ├── complex-mid-dark-random-test-s01-peak/      # [训练]
    │    │   ├── images/*.png          # 约 1300 张
    │    │   └── cvat_reviewed.xml
    │    ├── complex-mid-dark-random-val-s01-peak/       # [训练]
    │    │   ├── images/*.png          # 约 1500 张
    │    │   └── cvat_reviewed.xml
    │    └── complex-near-bright-random-test-s01-peak/   # [验证]
    │        ├── images/*.png          # 约 500 张
    │        └── cvat_reviewed.xml
    ├── HCFEvalSource/                 # 评估数据集（预留，当前未使用）
    └── HCFCVATTestSource/             # CVAT 标签导出测试数据集（预留）
```

## 模型架构

采用 MobileNetV3-Small，将第一层卷积从 RGB 三通道改造为单通道灰度输入（通过平均 ImageNet 预训练 RGB 权值实现），并将分类头替换为二分类输出（Left / Right）。

参数规模约 150 万，所有算子兼容 ONNX opset-13。

## 数据集划分策略

- **训练集**（6 个来源，约 4900 张）：除以下验证集外的全部数据来源
- **验证集**（2 个来源，约 1100 张）：
  - `complex-near-bright-random-test-s01-peak`（约 500 张）
  - `complex-mid-bright-random-val-s01-peak`（约 600 张）
- **测试集**：暂不划分。代码中保留了 Test DataLoader 和划分接口（通过配置中的 `test_sources` 和 `test_ratio` 控制），后续用户自行标注测试数据集后可直接使用
- 划分信息保存至 `outputs/splits.json`

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 查看数据集统计信息
python scripts/dataset_stats.py --config configs/hand_classifier.yaml

# 训练
python scripts/train.py --config configs/hand_classifier.yaml

# 评估
python scripts/evaluate.py --config configs/hand_classifier.yaml

# 导出 ONNX
python scripts/export_onnx.py --config configs/hand_classifier.yaml

# CVAT 标签导出测试
python scripts/cvat_label_test.py --config configs/hand_classifier.yaml

# 运行测试
python -m pytest tests/ -v
```

也可以使用 Makefile：

```bash
make setup           # 安装依赖
make dataset-stats   # 查看数据来源标签分布
make train           # 训练模型
make evaluate        # 评估模型
make export-onnx     # 导出 ONNX 模型
make cvat-label-test # 测试 CVAT 自动标注标签替换
make test            # 运行全部单元测试
make all             # test + train + evaluate + export-onnx + cvat-label-test
make clean           # 清理 outputs/、__pycache__/、.pytest_cache/
```

## 配置说明

编辑 `configs/hand_classifier.yaml` 可调整：

- 数据来源路径与划分比例
- 模型架构（`mobilenet_v3_small` / `mobilenet_v3_large`）
- 训练超参数（批次大小、学习率、训练轮数等）
- 数据增强策略
- ONNX 导出参数

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
