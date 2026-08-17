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

### 1. 核验数据集

```bash
python scripts/verify_datasets.py --config configs/train.yaml
```

输入：`autodl-tmp/DatasetFab/HCFTrainSource/*` + `HCFEvalSource/*` 下的数据来源目录
输出：每个数据来源的核验报告（结构、XML、标签分布、图像完整性）；退出码 0=正常，2=有错误

### 2. 查看数据集统计

```bash
python scripts/dataset_stats.py --config configs/train.yaml
```

输入：`autodl-tmp/DatasetFab/HCFTrainSource/*` 下的数据来源目录
输出：每个数据来源的标签分布（Left/Right 数量及比例）

### 3. 训练

```bash
python scripts/train.py --config configs/train.yaml
```

输入：HCFTrainSource 全部来源（含 eos_2.1 rain/thick 来源 + NegativeTrain 最新负样本池），约 29500 张；HCFEvalSource 验证集（含 NegativeVal 负样本验证来源），通过 glob 自动发现
输出：`<model_dir>/checkpoints/best.pth`、`<model_dir>/splits.json`、`<model_dir>/train/metrics.jsonl`，其中 `<model_dir>` = `paths.output_root/<model.version>/<model.architecture>/`（默认 `../autodl-tmp/TrainFab/outputs/v1/mobilenet_v3_small/`）

模型版本选择：修改 `configs/train.yaml` 中 `model.version`（`v1`/`v2`）与 `model.architecture`，例如 `version: "v2"` + `architecture: "v2_convnet_l"`。评估与导出配置中的版本/架构必须与训练一致；产物按系列/架构分目录，不同模型互不覆盖。

批量采样比例：`configs/train.yaml` 的 `sampling` 节控制每个训练 batch 的组成（no_hand_ratio 为无手样本占比，left_right_ratio 为有手样本中左右手占比），防止庞大的负样本池主导 hand presence 训练；仅作用于训练，验证集按真实分布使用。

多任务损失权重：总损失 = `training.handedness_loss_weight × loss_handedness + hand_presence.loss_weight × loss_hand_presence`，默认 1.2 : 1.0，handedness 略高，优先保证左右手精度；可调（如 1.5 进一步偏置）。

### 4. 评估

```bash
python scripts/evaluate.py --config configs/evaluate.yaml
```

输入：`<model_dir>/checkpoints/best.pth`（自动定位），验证集
输出：`<model_dir>/eval/val_metrics.json`

### 5. 导出 ONNX

```bash
python scripts/export_onnx.py --config configs/export_onnx.yaml
```

输入：`<model_dir>/checkpoints/best.pth`（自动定位）
输出：`<model_dir>/model.onnx`（v1.0/v2.0 输入输出接口一致）

### 6. CVAT 标签导出测试

```bash
python scripts/cvat_label_test.py --config configs/cvat_label_test.yaml
```

输入：`<model_dir>/model.onnx`（自动定位），CVAT 自动标注 XML + 图像
输出：`cvat_hcf.xml`（与输入 XML 同目录）

### 7. 推理：筛选负样本

```bash
python scripts/infer.py --config configs/infer.yaml
```

输入：`configs/infer.yaml` 指定的 ONNX 模型 + 图像目录
输出：低 hand_presence 图像拷贝至指定输出目录

### 8. 运行测试

```bash
python -m pytest tests/ -v
```

## Makefile 快捷命令

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
