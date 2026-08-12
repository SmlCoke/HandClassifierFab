# HCF QA 文档

本文档记录用户在本仓库使用过程中提出的问题及对应回答。

---

## 2026-08-12

### Q1：新增验证集时，是否只需要在服务器端 `~/autodl-tmp/DatasetFab/HCFEvalSource` 下按相同格式新增数据来源即可？

**A**：是的。`configs/train.yaml` 和 `configs/evaluate.yaml` 中的 `val_sources` 使用 glob 通配模式（`HCFEvalSource/*` 与 `HCFEvalSource/NegativeVal/*`），新增目录会被自动发现，无需修改配置。

注意两点：

1. 格式要求：`<来源名>/images/*.png` + `cvat_reviewed.xml`（或 `cvat_autolabel.xml`）。
2. 训练器通过**目录名**排除验证集来源（避免数据泄漏），因此验证来源的目录名不要与 `HCFTrainSource/` 下的任何训练来源重名。

---

### Q2：在训练集负样本目录（`~/autodl-tmp/DatasetFab/HCFTrainSource/NegativeTrain/`）中新增数据来源时，其中带有 `no_hand` 标签的 Hand ROI 样本是否会被当作负样本训练 hand presence？没有 `no_hand` 标签的样本是否会被当作正样本？

**A**：取决于新增来源**是否带 XML 标注文件**（逻辑见 `hand_classifier/parser.py`）：

| 情况 | 行为 |
|------|------|
| 带 XML（`cvat_reviewed.xml` / `cvat_autolabel.xml`） | 按 XML 中每个 `<image>` 的 `<tag>` 标签逐个判定 |
| ├ `no_hand` 标签 | → 负样本（presence=0），仅训练 hand presence |
| ├ `Left` / `Right` 标签 | → 正样本（presence=1），同时训练 handedness 和 hand presence |
| ├ `unknown_handedness` 标签 | → presence=1（正样本），handedness 忽略（-1） |
| └ `ignore_for_training` 标签 | → 完全排除，不参与训练 |
| 不带 XML（只有 `images/*.png`） | 全部图片视为负样本（presence=0） |

结论：文件夹名本身不强制任何标签，**XML 标签说了算**。若新增的是混合来源（同时含正/负样本），建议直接放在 `HCFTrainSource/` 下（而非 `NegativeTrain/` 下），并确保带有 XML 标注，否则全部图片会被当作 no_hand 负样本，反而污染 handedness 训练。

---

### Q3：`NegativeTrain/images/` 目录下残留的图片（各来源合并后的旧文件夹）是否会被当作负样本进入训练？

**A**：不会。parser 只处理"含 `images/` 子目录"或"含 XML"的目录。`NegativeTrain/images/` 内部没有嵌套的 `images/` 子目录也没有 XML，会被静默跳过（debug 日志）。已验证：删除该残留目录前后 `dataset-stats` 统计总数不变（no_hand=1233）。该残留目录已删除。

---

### Q4：想增加训练集数量，是否只需要在 `~/autodl-tmp/DatasetFab/HCFTrainSource` 下增加与现有数据来源格式相同的数据来源即可？

**A**：是的。`configs/train.yaml` 中 `train_sources` 使用 glob 模式（`HCFTrainSource/*` 与 `HCFTrainSource/NegativeTrain/*`），新增来源会被自动发现，无需修改代码或配置。格式要求：

- 标注来源：`<来源名>/images/*.png` + `cvat_reviewed.xml`
- 纯负样本来源（可选）：`<来源名>/images/*.png`（无 XML，全部视为 no_hand）

---

### Q5：双头模型训练时验证集 loss 一直是 NaN，导致早停失效、best checkpoint 选择失败，原因是什么？

**A**：根因是 `CrossEntropyLoss(weight=..., ignore_index=-1)` 在一个 batch 内**所有标签都被忽略时**（如验证集末尾 batch 全部是 no_hand 负样本，handedness 标签全为 -1），其内部分母为零，返回 NaN 而非 0。NaN 与 `inf` 比较结果为 False，导致 best checkpoint 永远不会被选中，早停计数器一直累加。

修复方式：计算 `loss_h` 前检查 batch 中是否存在有效标签，若全部为 -1 则直接置 `loss_h = 0`（见 `hand_classifier/trainer.py` 中的 guard 逻辑）。
