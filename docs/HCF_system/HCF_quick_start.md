# HCF 快速上手

手部左/右手二分类器流水线的纯命令参考。

## 环境准备

```bash
# 创建 conda 环境
conda create -n hand_classifier python=3.10
conda activate hand_classifier

# 安装 PyTorch（CUDA 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt
```

## 流水线

### 1. 查看数据集统计

```bash
python scripts/dataset_stats.py --config configs/train.yaml
```

输入：`autodl-tmp/DatasetFab/HCFTrainSource/*` 下的数据来源目录
输出：每个数据来源的标签分布（Left/Right 数量及比例）

### 2. 训练

```bash
python scripts/train.py --config configs/train.yaml
```

输入：6 个数据来源目录（训练集），约 4900 张图像；2 个数据来源目录（验证集），约 1100 张图像
输出：`outputs/checkpoints/best.pth`、`outputs/splits.json`、`outputs/train/metrics.jsonl`

### 3. 评估

```bash
python scripts/evaluate.py --config configs/evaluate.yaml
```

输入：`outputs/checkpoints/best.pth`，验证集
输出：`outputs/eval/val_metrics.json`

### 4. 导出 ONNX

```bash
python scripts/export_onnx.py --config configs/export_onnx.yaml
```

输入：`outputs/checkpoints/best.pth`
输出：`outputs/model.onnx`

### 5. CVAT 标签导出测试

```bash
python scripts/cvat_label_test.py --config configs/cvat_label_test.yaml
```

输入：`outputs/model.onnx`，CVAT 自动标注 XML + 图像
输出：`outputs/cvat_relabeled.xml`

### 6. 推理：筛选负样本

```bash
python scripts/infer.py --config configs/infer.yaml
```

输入：`configs/infer.yaml` 指定的 ONNX 模型 + 图像目录
输出：低 hand_presence 图像拷贝至指定输出目录

### 7. 运行测试

```bash
python -m pytest tests/ -v
```

## Makefile 快捷命令

```bash
make setup             # pip install -r requirements.txt
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
