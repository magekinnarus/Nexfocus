"""Tracked W11a GAN upscale worker coverage."""

import os
import sys
import numpy as np
import pytest
import torch
from types import SimpleNamespace

# Setup sys.path
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import backend.auxiliary_workers.gan_upscale_worker as gan_worker_module
from backend.auxiliary_workers.execution import active_auxiliary_worker
from backend.auxiliary_workers.gan_upscale_worker import GanUpscaleWorker, run_gan_upscale
from backend.auxiliary_workers.telemetry import telemetry_sink
import ldm_patched.pfn.model_loading as loading


def test_gan_upscale_worker_lifecycle(monkeypatch) -> None:
    """Test GanUpscaleWorker explicit load, infer, and teardown phases with telemetry events."""
    # 1. Prepare mocks
    mock_model_name = "mock_model.pth"
    dummy_input = np.zeros((128, 128, 3), dtype=np.uint8)
    dummy_output = np.ones((512, 512, 3), dtype=np.uint8)

    # Mock path check
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    # Mock torch load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"state_dict": {}})

    # Mock Spandrel model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 4
            self.tags = ["RGB"]
            self.architecture = SimpleNamespace(id="RealESRGANv2")
            # Mock self.model parameter dtype detection
            self.model = SimpleNamespace(parameters=lambda: iter([torch.tensor(0.0, dtype=torch.float32)]))

        def float(self):
            return self

        def cpu(self):
            return self

        def to(self, device):
            return self

        def __call__(self, x):
            return x

    monkeypatch.setattr(loading, "load_state_dict", lambda sd: MockModel())

    # Mock NexUpscaleEngine
    class MockNexUpscaleEngine:
        def process(self, img, upscale_fn, native_scale, device, is_bgr=True, dtype=None):
            return dummy_output

    monkeypatch.setattr("backend.auxiliary_workers.gan_upscale_worker.NexUpscaleEngine", MockNexUpscaleEngine)

    # Mock cv2.resize
    import cv2
    monkeypatch.setattr(cv2, "resize", lambda img, size, interpolation: np.ones((size[1], size[0], 3), dtype=np.uint8))

    # Spy telemetry
    snapshots = []
    # Instantiate
    worker = GanUpscaleWorker()
    assert worker.model is None
    assert worker.model_name is None

    # Test Load
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        worker.load(mock_model_name)

    assert worker.model is not None
    assert worker.model_name == mock_model_name
    assert any(s["event"] == "gan_upscale_worker_load_begin" for s in snapshots)
    assert any(s["event"] == "gan_upscale_worker_load_complete" for s in snapshots)

    # Test Infer (Native scale)
    snapshots.clear()
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        result = worker.infer(dummy_input)

    assert result.shape == (512, 512, 3)
    assert any(s["event"] == "gan_upscale_worker_infer_begin" for s in snapshots)
    assert any(s["event"] == "gan_upscale_worker_infer_complete" for s in snapshots)

    # Test Infer (Overridden scale)
    snapshots.clear()
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        result_override = worker.infer(dummy_input, scale_override=2.0)

    assert result_override.shape == (256, 256, 3)

    # Test Teardown
    snapshots.clear()
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        worker.teardown()

    assert worker.model is None
    assert worker.model_name is None
    assert any(s["event"] == "gan_upscale_worker_teardown_begin" for s in snapshots)
    assert any(s["event"] == "gan_upscale_worker_teardown_complete" for s in snapshots)


@pytest.mark.parametrize("failure_phase", ["load", "infer"])
def test_run_gan_upscale_releases_lease_and_worker_on_failure(monkeypatch, failure_phase) -> None:
    calls = []

    class FailingWorker:
        def load(self, model_name):
            calls.append(("load", model_name))
            if failure_phase == "load":
                raise RuntimeError("synthetic load failure")

        def infer(self, img, scale_override=None):
            calls.append(("infer", scale_override))
            raise RuntimeError("synthetic inference failure")

        def teardown(self):
            calls.append("teardown")

    monkeypatch.setattr(gan_worker_module, "GanUpscaleWorker", FailingWorker)
    snapshots = []
    failure_message = "synthetic load failure" if failure_phase == "load" else "synthetic inference failure"
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        with pytest.raises(RuntimeError, match=failure_message):
            run_gan_upscale(
                np.zeros((8, 8, 3), dtype=np.uint8),
                model_name="mock_model.pth",
            )

    expected_calls = [("load", "mock_model.pth")]
    if failure_phase == "infer":
        expected_calls.append(("infer", None))
    expected_calls.append("teardown")
    assert calls == expected_calls
    assert active_auxiliary_worker() is None
    assert any(s["event"] == "auxiliary_execution_failed" for s in snapshots)
    assert any(s["event"] == "auxiliary_execution_released" for s in snapshots)


def test_upscaler_scale_probe_retains_metadata_only(monkeypatch, tmp_path) -> None:
    import modules.upscaler as upscaler

    model_path = tmp_path / "mock_model.pth"
    model_path.write_bytes(b"metadata-key")
    calls = []

    class MetadataWorker:
        def load(self, model_name):
            calls.append(("load", model_name))

        def get_native_scale(self):
            calls.append("scale")
            return 4

        def teardown(self):
            calls.append("teardown")

    upscaler.clear_model_cache()
    monkeypatch.setattr(upscaler, "_resolve_model_path", lambda _name: model_path)
    monkeypatch.setattr(gan_worker_module, "GanUpscaleWorker", MetadataWorker)

    assert upscaler.get_model_scale_for_name("mock_model.pth") == 4
    assert upscaler.get_model_scale_for_name("mock_model.pth") == 4
    assert calls == [("load", "mock_model.pth"), "scale", "teardown"]
    assert not hasattr(upscaler, "_cached_model")
    assert list(upscaler._MODEL_SCALE_METADATA_CACHE.values()) == [4]
