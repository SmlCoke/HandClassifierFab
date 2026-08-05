"""Tests for CVAT XML parser."""

import os
import pytest

from hand_classifier.parser import parse_cvat_xml, collect_all_samples


def test_parse_reviewed_xml(example_xml, example_images):
    """Parse cvat_reviewed.xml (gold standard) and verify samples."""
    samples = parse_cvat_xml(example_xml, example_images)

    assert len(samples) > 0, "Should parse at least one sample"
    for s in samples:
        assert "image_path" in s
        assert "label" in s
        assert os.path.exists(s["image_path"]), f"Image should exist: {s['image_path']}"
        assert s["label"] in (0, 1), f"Label should be 0 or 1, got {s['label']}"

    # Check exclude behavior: no ignore_for_training samples
    labels = [s["label"] for s in samples]
    assert all(l in (0, 1) for l in labels)


def test_parse_autolabel_xml(test_xml, test_images):
    """Parse cvat_autolabel.xml (all unknown_handedness, should return empty)."""
    samples = parse_cvat_xml(test_xml, test_images)
    # All should have unknown_handedness → excluded
    assert len(samples) == 0, (
        "All autolabel samples should be excluded (unknown_handedness)"
    )


def test_parse_nonexistent_xml():
    """Gracefully handle non-existent XML file."""
    with pytest.raises(FileNotFoundError):
        parse_cvat_xml("/nonexistent/path.xml", "/nonexistent/images")


def test_parse_missing_images(example_xml, tmp_path):
    """All images missing should result in zero samples."""
    # tmp_path has no images
    samples = parse_cvat_xml(example_xml, str(tmp_path))
    assert len(samples) == 0


def test_collect_all_samples_example(example_dir):
    """Collect samples from the example dataset directory."""
    samples = collect_all_samples([example_dir])
    assert len(samples) > 0
    for s in samples:
        assert "source" in s
        assert s["source"] == "dataset1"


def test_collect_all_samples_nonexistent():
    """Non-existent directories should be skipped with warning."""
    samples = collect_all_samples(["/nonexistent/path"])
    assert len(samples) == 0


def test_collect_all_samples_autolabel(test_dir):
    """Autolabel source should return 0 trainable samples."""
    samples = collect_all_samples([test_dir])
    assert len(samples) == 0
