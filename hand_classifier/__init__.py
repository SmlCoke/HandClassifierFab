"""Hand Left/Right Binary Classifier - Core Package (dual-head version)."""

from hand_classifier.config import load_config
from hand_classifier.parser import (
    parse_cvat_xml, collect_all_samples, collect_negative_samples,
)
from hand_classifier.dataset import (
    HandROIDataset, get_transforms, split_dataset,
    save_split_info, compute_class_weights,
)
from hand_classifier.trainer import train
from hand_classifier.evaluator import evaluate
from hand_classifier.exporter import export_onnx
from hand_classifier.relabel import relabel_cvat_xml, compute_agreement

__all__ = [
    "load_config",
    "parse_cvat_xml",
    "collect_all_samples",
    "collect_negative_samples",
    "HandROIDataset",
    "get_transforms",
    "split_dataset",
    "save_split_info",
    "compute_class_weights",
    "train",
    "evaluate",
    "export_onnx",
    "relabel_cvat_xml",
    "compute_agreement",
]
