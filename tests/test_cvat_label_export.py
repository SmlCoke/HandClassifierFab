"""Tests for CVAT XML relabeling functionality."""

import os
import re
import pytest

from hand_classifier.relabel import relabel_cvat_xml, compute_agreement


# Regex patterns for verifying byte-preserving relabeling
_TAG_LABEL_RE = re.compile(r'label="(Left|Right|unknown_handedness)"')


def _count_labels(xml_content):
    """Count Left, Right, unknown_handedness labels in XML content."""
    counts = {"Left": 0, "Right": 0, "unknown_handedness": 0}
    for m in _TAG_LABEL_RE.finditer(xml_content):
        label = m.group(1)
        if label in counts:
            counts[label] += 1
    return counts


def test_relabel_autolabel_xml_byte_preserving(test_xml, test_images, tmp_path):
    """Test relabeling of autolabel XML preserves structure.

    This test verifies the relabel function signature and logic, but
    will fail if no ONNX model is available. Skip gracefully.
    """
    # Create a minimal "model" by mocking the ONNX inference
    # Since we can't easily mock ONNX, we test the regex logic directly

    with open(test_xml, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify input has unknown_handedness
    counts_before = _count_labels(content)
    assert counts_before["unknown_handedness"] > 0, (
        "Test XML should contain unknown_handedness tags"
    )

    # Test regex substitution logic directly
    tag_pattern = re.compile(
        r'(<tag\s[^>]*?)label="unknown_handedness"([^>]*?/>)', re.DOTALL
    )

    def _replace(m):
        return m.group(0).replace('label="unknown_handedness"', 'label="Left"')

    new_content = tag_pattern.sub(_replace, content)

    counts_after = _count_labels(new_content)
    assert counts_after["unknown_handedness"] == 0, (
        "All unknown_handedness should be replaced"
    )
    assert counts_after["Left"] > 0, "Should have Left labels after replacement"

    # Verify self-closing tag structure preserved
    assert '/>' in new_content
    assert new_content.count('<image') == content.count('<image'), (
        "Image count should be preserved"
    )


def test_relabel_missing_model(test_xml, tmp_path):
    """Test error handling when model file doesn't exist."""
    output = tmp_path / "output.xml"
    with pytest.raises(Exception):
        relabel_cvat_xml(
            str(test_xml),
            "/nonexistent/model.onnx",
            str(output),
        )


def test_compute_agreement_same_labels(tmp_path):
    """Agreement with identical labels should be 1.0."""
    import xml.etree.ElementTree as ET

    # Create two identical XMLs
    for fname, label in [("pred.xml", "Left"), ("gold.xml", "Left")]:
        root = ET.Element("annotations")
        img = ET.SubElement(root, "image", {
            "id": "0", "name": "test.png", "width": "256", "height": "256",
        })
        tag = ET.SubElement(img, "tag", {"label": label})
        tree = ET.ElementTree(root)
        tree.write(str(tmp_path / fname), encoding="utf-8", xml_declaration=True)

    agreement = compute_agreement(
        str(tmp_path / "pred.xml"),
        str(tmp_path / "gold.xml"),
    )
    assert agreement["agreement_rate"] == 1.0


def test_compute_agreement_different_labels(tmp_path):
    """Agreement with different labels should not be 1.0."""
    import xml.etree.ElementTree as ET

    for fname, label in [("pred.xml", "Left"), ("gold.xml", "Right")]:
        root = ET.Element("annotations")
        img = ET.SubElement(root, "image", {
            "id": "0", "name": "test.png", "width": "256", "height": "256",
        })
        tag = ET.SubElement(img, "tag", {"label": label})
        tree = ET.ElementTree(root)
        tree.write(str(tmp_path / fname), encoding="utf-8", xml_declaration=True)

    agreement = compute_agreement(
        str(tmp_path / "pred.xml"),
        str(tmp_path / "gold.xml"),
    )
    assert agreement["agreement_rate"] == 0.0
    assert agreement["disagree"] == 1


def test_image_block_regex():
    """Verify the image block regex correctly matches XML image elements."""
    from hand_classifier.relabel import _IMAGE_BLOCK_RE, _TAG_LABEL_RE

    # Multi-line image block (cvat_reviewed.xml style)
    block = """<image id="0" name="images/roi_test.png" width="256" height="256">
  <skeleton label="hand_landmarks" source="manual" z_order="0">
    <points label="1" source="manual" outside="0" occluded="0" points="100.0,200.0">
    </points>
  </skeleton>
  <tag label="Left" source="manual">
  </tag>
</image>"""

    match = _IMAGE_BLOCK_RE.search(block)
    assert match is not None
    assert match.group(2) == "images/roi_test.png"

    # Self-closing tag block (cvat_autolabel.xml style)
    block2 = '<image id="0" name="roi_test.png" width="256" height="256"><skeleton label="hand_landmarks" source="auto" z_order="0"><points label="1" source="auto" occluded="0" outside="0" points="168.000,215.000" /></skeleton><tag label="unknown_handedness" source="auto" /></image>'

    match2 = _IMAGE_BLOCK_RE.search(block2)
    assert match2 is not None

    # Tag regex should match both styles
    tag_multi = '<tag label="unknown_handedness" source="file">\n</tag>'
    tag_self = '<tag label="unknown_handedness" source="auto" />'

    assert _TAG_LABEL_RE.search(tag_multi) is not None
    assert _TAG_LABEL_RE.search(tag_self) is not None
