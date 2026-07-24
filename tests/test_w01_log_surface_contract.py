import ast
import logging
from pathlib import Path
from types import SimpleNamespace

from backend.hardware import _emit_residency_log


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _normal_print_surfaces(source: str) -> list[str]:
    """Return direct and simple indirect expressions exposed through print()."""
    tree = ast.parse(source)
    assigned_values: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        value = None
        targets = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]

        if value is None:
            continue
        rendered_value = ast.unparse(value)
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    assigned_values.setdefault(name.id, []).append(rendered_value)

    surfaces: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            continue

        rendered = [ast.unparse(arg) for arg in node.args]
        referenced_names = {
            name.id
            for arg in node.args
            for name in ast.walk(arg)
            if isinstance(name, ast.Name)
        }
        for name in referenced_names:
            rendered.extend(assigned_values.get(name, ()))
        surfaces.append(" ".join(rendered))
    return surfaces


def test_print_surface_detector_catches_multiline_and_indirect_markers():
    surfaces = _normal_print_surfaces(
        """
message = "[Internal] indirect"
print(message)
print(
    f"[Internal] multiline"
)
"""
    )
    assert len(surfaces) == 2
    assert all("[Internal]" in surface for surface in surfaces)


def test_residency_telemetry_is_debug_only(caplog):
    plan = SimpleNamespace(
        notes={"profile": "test", "phase": "test"},
        pinned=("vae",),
        warm=("unet",),
        evictable=("clip",),
    )

    with caplog.at_level(logging.INFO):
        _emit_residency_log("test", plan=plan)
    assert "[Nex-Residency]" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _emit_residency_log("test", plan=plan)
    assert "[Nex-Residency]" in caplog.text


def test_internal_telemetry_has_no_normal_mode_print_surface():
    source_expectations = {
        "modules/pipeline/stage_runtime.py": ("[Residency]",),
        "modules/async_worker.py": ("[Route]",),
        "backend/legacy_governor.py": ("[Nex-Perf]",),
        "backend/sampling.py": ("[Nex-Perf]",),
        "modules/core.py": ("[Nex-Perf]",),
        "backend/sdxl_assembly/gateway.py": (
            "[SDXL RUN BEGIN]",
            "[SDXL RUN END]",
            "[SDXL LORA ADMISSION]",
        ),
        "backend/sdxl_assembly/lifecycle_coordinator.py": ("[SDXL LIFECYCLE RELEASE",),
        "backend/sdxl_assembly/request_builder.py": ("[SDXL LORA ADMISSION]",),
        "modules/upscale_engine.py": ("[Nex-Engine]",),
    }

    for relative_path, markers in source_expectations.items():
        source = _source(relative_path)
        print_surfaces = _normal_print_surfaces(source)
        for marker in markers:
            assert all(marker not in surface for surface in print_surfaces), (
                f"{relative_path} exposes internal marker {marker!r} through print()"
            )


def test_output_format_change_updates_history_link_without_dangling_return():
    source = _source("modules/ui_logic.py")
    assert "output_format.input(lambda x: gr.update(output_format=x)" not in source
    assert "output_format.change(" in source
    assert "inputs=[output_format]" in source
    assert "outputs=[history_link]" in source
