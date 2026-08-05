"""Shared test fixtures."""

import os
import sys
import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXAMPLE_DIR = os.path.join(DATA_DIR, "examples", "dataset1")
TEST_DIR = os.path.join(DATA_DIR, "dataset_test", "complex-near-bright-random-val-s01-peak")


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def example_dir():
    return EXAMPLE_DIR


@pytest.fixture
def example_xml():
    return os.path.join(EXAMPLE_DIR, "cvat_reviewed.xml")


@pytest.fixture
def example_images():
    return os.path.join(EXAMPLE_DIR, "images")


@pytest.fixture
def test_dir():
    return TEST_DIR


@pytest.fixture
def test_xml():
    return os.path.join(TEST_DIR, "cvat_autolabel.xml")


@pytest.fixture
def test_images():
    return os.path.join(TEST_DIR, "images")
