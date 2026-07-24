from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REQUIRED_SPECS = {
    "transformers": SpecifierSet(">=4.42.4"),
    "huggingface-hub": SpecifierSet(">=0.32,<1.0"),
    "tokenizers": SpecifierSet("==0.19.1"),
    "accelerate": SpecifierSet(">=0.32.1"),
}

VALIDATED_BASELINE = {
    "transformers": "4.44.2",
    "huggingface-hub": "0.36.2",
    "tokenizers": "0.19.1",
    "accelerate": "1.13.0",
}


def test_venv_interpreter():
    # Check if active interpreter is in the repo venv
    assert "venv" in Path(sys.executable).parts, (
        f"Active interpreter does not appear to be the repo venv. "
        f"Path: {sys.executable}. Recommended: .\\venv\\Scripts\\python.exe"
    )


@pytest.mark.parametrize("package_name", list(REQUIRED_SPECS.keys()))
def test_package_version_specs(package_name):
    # Read version
    try:
        version_text = metadata.version(package_name)
    except metadata.PackageNotFoundError as exc:
        raise AssertionError(f"{package_name} is not installed in the active interpreter.") from exc

    # Check version matches specifier set
    version = Version(version_text)
    spec = REQUIRED_SPECS[package_name]
    assert version in spec, f"{package_name}=={version_text} does not satisfy {spec}"


def test_transformers_import_succeeds():
    try:
        importlib.import_module("transformers")
    except Exception as exc:
        raise AssertionError(f"transformers import failed: {exc}") from exc
