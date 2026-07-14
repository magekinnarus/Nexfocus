from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


REQUIRED_SPECS = {
    "transformers": SpecifierSet(">=4.42.4"),
    "huggingface-hub": SpecifierSet(">=0.32,<1.0"),
    "hf-xet": SpecifierSet(">=1.1.3,<2.0"),
    "tokenizers": SpecifierSet("==0.19.1"),
    "accelerate": SpecifierSet(">=0.32.1"),
}

VALIDATED_BASELINE = {
    "transformers": "4.44.2",
    "huggingface-hub": "0.36.2",
    "hf-xet": "1.5.1",
    "tokenizers": "0.19.1",
    "accelerate": "1.13.0",
}


def _read_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{package_name} is not installed in the active interpreter.") from exc


def _check_spec(package_name: str, version_text: str) -> tuple[bool, str]:
    try:
        version = Version(version_text)
    except InvalidVersion:
        return False, f"{package_name} has an unparseable version: {version_text!r}"

    spec = REQUIRED_SPECS[package_name]
    if version in spec:
        return True, ""
    return False, f"{package_name}=={version} does not satisfy {spec}"


def main() -> int:
    print(f"Interpreter: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")

    if "venv" not in Path(sys.executable).parts:
        print("WARNING: active interpreter does not appear to be the repo venv.")
        print("Recommended interpreter: .\\venv\\Scripts\\python.exe")

    failures: list[str] = []
    observed_versions: dict[str, str] = {}

    for package_name in REQUIRED_SPECS:
        try:
            version_text = _read_version(package_name)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue

        observed_versions[package_name] = version_text
        ok, message = _check_spec(package_name, version_text)
        status = "OK" if ok else "FAIL"
        baseline = VALIDATED_BASELINE[package_name]
        print(
            f"{status:<4} {package_name}=={version_text} "
            f"(required {REQUIRED_SPECS[package_name]}, validated baseline {baseline})"
        )
        if not ok:
            failures.append(message)

    try:
        importlib.import_module("transformers")
        print("OK   transformers import succeeded")
    except Exception as exc:  # pragma: no cover - exercised via manual validation
        failures.append(f"transformers import failed: {exc}")
        print(f"FAIL transformers import failed: {exc}")

    if failures:
        print("\nValidation environment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation environment matches the W12 runtime baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
