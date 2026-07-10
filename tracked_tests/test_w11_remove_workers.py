"""Tracked W11b lifecycle, route, and Flux-boundary coverage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import backend.auxiliary_workers.background_removal_worker as bgr_worker_module
import backend.auxiliary_workers.mat_inpaint_worker as mat_worker_module
from backend.auxiliary_workers.execution import active_auxiliary_worker
from backend.auxiliary_workers.telemetry import telemetry_sink
from backend.auxiliary_workers.background_removal_worker import (
    BackgroundRemovalWorker,
    run_background_removal,
)
from backend.auxiliary_workers.mat_inpaint_worker import MatInpaintWorker, run_mat_inpaint


class _DummyRemover:
    def __init__(self, jit, ckpt):
        self.jit = jit
        self.ckpt = ckpt

    def process(self, image, *, type, threshold):
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = 17
        rgba[1:-1, 1:-1, 3] = 255
        from PIL import Image

        return Image.fromarray(rgba, mode="RGBA")


class _DummyMat:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def to(self, target):
        self.calls.append(("to", str(target)))
        return self

    def __call__(self, image, mask):
        self.calls.append(("infer", tuple(image.shape)))
        return torch.zeros_like(image)


def test_background_worker_returns_rgba_and_binary_mask(monkeypatch):
    monkeypatch.setattr(bgr_worker_module.model_registry, "ensure_asset", lambda *args, **kwargs: "bgr.ckpt")
    monkeypatch.setattr(bgr_worker_module, "Remover", _DummyRemover)
    snapshots = []
    worker = BackgroundRemovalWorker()

    with telemetry_sink(snapshots.append):
        worker.load(jit=False)
        rgba, mask = worker.infer(np.zeros((4, 5, 3), dtype=np.uint8), threshold=0.4)
        worker.teardown()

    assert rgba.shape == (4, 5, 4)
    assert rgba.dtype == np.uint8
    assert mask.shape == (4, 5)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert worker.remover is None
    assert any(item["event"] == "background_removal_worker_load_complete" for item in snapshots)
    assert any(item["event"] == "background_removal_worker_teardown_complete" for item in snapshots)


@pytest.mark.parametrize("worker_kind", ["background", "mat"])
def test_remove_worker_failure_releases_worker_and_lease(monkeypatch, worker_kind):
    calls = []

    if worker_kind == "background":
        class FailingWorker:
            def load(self, *, jit=True):
                calls.append(("load", jit))
                raise RuntimeError("synthetic load failure")

            def teardown(self):
                calls.append("teardown")

        monkeypatch.setattr(bgr_worker_module, "BackgroundRemovalWorker", FailingWorker)
        runner = lambda: run_background_removal(np.zeros((4, 4, 3), dtype=np.uint8), jit=False)
    else:
        class FailingWorker:
            def load(self, *, model_name):
                calls.append(("load", model_name))
                raise RuntimeError("synthetic load failure")

            def teardown(self):
                calls.append("teardown")

        monkeypatch.setattr(mat_worker_module, "MatInpaintWorker", FailingWorker)
        runner = lambda: run_mat_inpaint(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        )

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        runner()

    assert calls[-1] == "teardown"
    assert active_auxiliary_worker() is None


@pytest.mark.parametrize("worker_kind", ["background", "mat"])
def test_remove_worker_inference_failure_releases_worker_and_lease(monkeypatch, worker_kind):
    calls = []

    if worker_kind == "background":
        class FailingWorker:
            def load(self, *, jit=True):
                calls.append("load")

            def infer(self, image, *, threshold=0.5):
                calls.append("infer")
                raise RuntimeError("synthetic inference failure")

            def teardown(self):
                calls.append("teardown")

        monkeypatch.setattr(bgr_worker_module, "BackgroundRemovalWorker", FailingWorker)
        runner = lambda: run_background_removal(np.zeros((4, 4, 3), dtype=np.uint8))
    else:
        class FailingWorker:
            def load(self, *, model_name):
                calls.append("load")

            def infer(self, image, mask, *, seed=0, mask_dilate=16):
                calls.append("infer")
                raise RuntimeError("synthetic inference failure")

            def teardown(self):
                calls.append("teardown")

        monkeypatch.setattr(mat_worker_module, "MatInpaintWorker", FailingWorker)
        runner = lambda: run_mat_inpaint(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        )

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        runner()

    assert calls == ["load", "infer", "teardown"]
    assert active_auxiliary_worker() is None


def test_mat_worker_preserves_small_and_tiled_output_contracts():
    calls = []
    worker = MatInpaintWorker()
    worker.model = _DummyMat(calls)

    small = np.full((32, 40, 3), 120, dtype=np.uint8)
    small_mask = np.zeros((32, 40), dtype=np.uint8)
    small_mask[8:16, 12:20] = 255
    small_result = worker.infer(small, small_mask, seed=11, mask_dilate=0)
    assert small_result.shape == small.shape
    assert small_result.dtype == np.uint8

    calls.clear()
    large = np.full((513, 513, 3), 120, dtype=np.uint8)
    large_mask = np.zeros((513, 513), dtype=np.uint8)
    large_mask[8:16, 12:20] = 255
    large_result = worker.infer(large, large_mask, seed=11, mask_dilate=0)
    assert large_result.shape == large.shape
    assert large_result.dtype == np.uint8
    assert sum(1 for item in calls if item[0] == "infer") > 0
    worker.teardown()


def test_combined_auxiliary_helpers_use_sequential_leases(monkeypatch):
    events = []

    class BgrWorker:
        def load(self, *, jit=True):
            events.append(("bgr_load", active_auxiliary_worker()))

        def infer(self, image, *, threshold=0.5):
            events.append(("bgr_infer", active_auxiliary_worker()))
            return np.zeros((4, 4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)

        def teardown(self):
            events.append(("bgr_teardown", active_auxiliary_worker()))

    class MatWorker:
        def load(self, *, model_name):
            events.append(("mat_load", active_auxiliary_worker()))

        def infer(self, image, mask, *, seed=0, mask_dilate=16):
            events.append(("mat_infer", active_auxiliary_worker()))
            return np.zeros_like(image)

        def teardown(self):
            events.append(("mat_teardown", active_auxiliary_worker()))

    monkeypatch.setattr(bgr_worker_module, "BackgroundRemovalWorker", BgrWorker)
    monkeypatch.setattr(mat_worker_module, "MatInpaintWorker", MatWorker)

    run_background_removal(np.zeros((4, 4, 3), dtype=np.uint8))
    assert active_auxiliary_worker() is None
    run_mat_inpaint(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    assert active_auxiliary_worker() is None

    assert [name for name, _active in events] == [
        "bgr_load", "bgr_infer", "bgr_teardown", "mat_load", "mat_infer", "mat_teardown"
    ]
    assert all(active == "background_removal" for _name, active in events[:3])
    assert all(active == "mat_inpaint" for _name, active in events[3:])


def test_removal_file_loader_preserves_legacy_rgba_compositing(tmp_path):
    from PIL import Image
    import modules.pipeline.routes as routes

    image_path = tmp_path / "transparent-input.png"
    Image.new("RGBA", (1, 1), (10, 20, 30, 0)).save(image_path)

    loaded = routes._load_removal_array(str(image_path), mode="RGB")

    assert loaded.shape == (1, 1, 3)
    assert loaded.dtype == np.uint8
    assert loaded[0, 0].tolist() == [255, 255, 255]


def test_removal_stage_dispatches_workers_without_auxiliary_preflight_cleanup(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    from modules.flux_fill_surface import OBJR_ENGINE_MAT
    from types import SimpleNamespace

    events = []
    persisted = []
    yielded = []
    progress = []
    cleanup_calls = []

    task_state = SimpleNamespace(
        goals=[flags.remove_bg, flags.remove_obj],
        remove_base_image="base.png",
        remove_mask_image="initial-mask.png",
        seed=7,
        steps=24,
        sampler_name="euler",
        scheduler_name="normal",
        objr_mask_dilate=16,
        objr_engine=OBJR_ENGINE_MAT,
        bgr_threshold=0.4,
        bgr_jit=True,
        inpaint_context="stale",
    )
    context = SimpleNamespace(
        task_state=task_state,
        progressbar_callback=lambda task, pct, message: progress.append((pct, message)),
        yield_result_callback=lambda task, paths, pct, do_not_show_finished_images=False: yielded.append((paths, pct)),
    )

    monkeypatch.setattr(routes, "_load_removal_array", lambda path, mode: np.zeros((4, 4, 3), dtype=np.uint8) if mode == "RGB" else np.zeros((4, 4), dtype=np.uint8))

    def fake_bgr(image, *, threshold, jit):
        events.append("bgr")
        return np.zeros((4, 4, 4), dtype=np.uint8), np.full((4, 4), 255, dtype=np.uint8)

    def fake_mat(image, mask, *, seed, mask_dilate):
        events.append("mat")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("backend.auxiliary_workers.run_background_removal", fake_bgr)
    monkeypatch.setattr("backend.auxiliary_workers.run_mat_inpaint", fake_mat)
    monkeypatch.setattr(routes, "_save_removal_temp", lambda payload: "bgr-mask.png" if np.asarray(payload).ndim == 2 else "bgr-rgba.png" if np.asarray(payload).shape[-1] == 4 else "mat.png")
    monkeypatch.setattr(routes, "_save_logged_output", lambda context, payload, description, **kwargs: persisted.append((payload, description)) or f"saved::{description}")
    monkeypatch.setattr("backend.resources.begin_memory_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.resources.end_memory_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.resources.cleanup_memory", lambda *args, **kwargs: cleanup_calls.append(args[0]))

    result = routes.RemovalStage().execute(context)

    assert result.route_complete is True
    assert events == ["bgr", "mat"]
    assert task_state.remove_mask_image == "bgr-mask.png"
    assert cleanup_calls == []
    assert progress == [
        (5, "Preparing Auxiliary Removal..."),
        (10, "Background Removal Starting..."),
        (60, "Object Removal Starting..."),
    ]
    assert all("Flux Fill" not in message for _pct, message in progress)
    assert yielded == [
        (["saved::Background Removal Subject", "saved::Background Removal Mask"], 50),
        (["saved::Object Removal"], 100),
    ]
    assert persisted[-1] == ("mat.png", "Object Removal")


def test_removal_stage_keeps_flux_on_flux_fill_adapter_boundary():
    import modules.pipeline.routes as routes
    resources = routes.RemovalStage().describe_resources(type("Context", (), {})())
    owners = {resource.resource_id: resource.owner for resource in resources}
    assert owners["background_removal_worker"].startswith("backend.auxiliary_workers")
    assert owners["mat_inpaint_worker"].startswith("backend.auxiliary_workers")


def test_compatibility_bridges_have_no_live_model_cache_authority():
    import modules.bgr_engine as bgr_engine
    import modules.objr_engine as objr_engine

    assert not hasattr(bgr_engine, "_remover_instance")
    assert not hasattr(bgr_engine, "_cached_jit")
    assert not hasattr(objr_engine, "_model_instance")
    assert bgr_engine.unload_model() is None
    assert objr_engine.unload_model() is None
