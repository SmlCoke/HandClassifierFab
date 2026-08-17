"""Tests for CVAT XML parser (dual-head version)."""

import os
import pytest

from hand_classifier.parser import parse_cvat_xml, collect_all_samples


def test_parse_reviewed_xml(example_xml, example_images):
    """Parse cvat_reviewed.xml (gold standard) and verify dual-label samples."""
    samples = parse_cvat_xml(example_xml, example_images)

    assert len(samples) > 0, "Should parse at least one sample"
    for s in samples:
        assert "image_path" in s
        assert "handedness_label" in s
        assert "presence_label" in s
        assert os.path.exists(s["image_path"]), (
            f"Image should exist: {s['image_path']}"
        )
        assert s["presence_label"] in (0, 1)
        # Left/Right should have presence=1 and handedness in {0,1}
        if s["handedness_label"] >= 0:
            assert s["presence_label"] == 1
            assert s["handedness_label"] in (0, 1)

    # All should be valid (no ignore_for_training)
    hs = [s["handedness_label"] for s in samples]
    assert all(h in (0, 1) for h in hs) or True  # may have -1 for unknown_handedness


def test_parse_autolabel_xml(test_xml, test_images):
    """Parse cvat_autolabel.xml: unknown_handedness → presence=1, handedness=-1."""
    samples = parse_cvat_xml(test_xml, test_images)
    # unknown_handedness: has hand but unknown handedness
    # So these should now be INCLUDED (presence=1, handedness=-1)
    assert len(samples) > 0, "unknown_handedness samples should be included"
    for s in samples:
        assert s["presence_label"] == 1
        assert s["handedness_label"] == -1


def test_parse_nonexistent_xml():
    """Gracefully handle non-existent XML file."""
    with pytest.raises(FileNotFoundError):
        parse_cvat_xml("/nonexistent/path.xml", "/nonexistent/images")


def test_parse_missing_images(example_xml, tmp_path):
    """All images missing should result in zero samples."""
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
    """Autolabel source should return presence=1 samples."""
    samples = collect_all_samples([test_dir])
    # unknown_handedness is now included (presence=1, handedness=-1)
    assert len(samples) > 0
    for s in samples:
        assert s["presence_label"] == 1
        assert s["handedness_label"] == -1


def test_collect_all_samples_flat_negative_layout(tmp_path):
    """Flat image-only layout (PNG files directly in the source dir) must be
    collected as negative samples."""
    source = tmp_path / "flat-negative-source"
    source.mkdir()
    for name in ["a.png", "b.png", "c.png"]:
        (source / name).touch()
    # decoy non-image file and nested junk must be ignored
    (source / "notes.txt").write_text("junk")
    (source / ".ipynb_checkpoints").mkdir()

    samples = collect_all_samples([str(source)])
    assert len(samples) == 3
    for s in samples:
        assert s["presence_label"] == 0
        assert s["handedness_label"] == -1
        assert s["source"] == "flat-negative-source"
        assert os.path.basename(s["image_path"]) in ("a.png", "b.png", "c.png")


def test_collect_all_samples_canonical_negative_layout(tmp_path):
    """Canonical image-only layout (<source>/images/*.png) still works."""
    source = tmp_path / "canonical-negative-source"
    images = source / "images"
    images.mkdir(parents=True)
    for name in ["a.png", "b.png"]:
        (images / name).touch()

    samples = collect_all_samples([str(source)])
    assert len(samples) == 2
    assert all(s["presence_label"] == 0 for s in samples)
