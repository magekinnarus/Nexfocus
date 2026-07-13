from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import torch

from backend.sdxl_assembly.contracts import (
    ResolvedFileIdentity,
    SDXLAssemblyRequest,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SDXLAssemblyValidationError,
    SDXLAssemblyEligibilityError,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
)
from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
from backend.sdxl_assembly.lifecycle_coordinator import (
    release_domains,
    LifecycleDomain,
)
from modules.task_state import TaskState
from modules.parameter_registry import _normalize_sdxl_assembly_posture_value

def _identity(name: str, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=Path(name),
        sha256=sha,
        size_bytes=1,
        modified_ns=1,
    )

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    # Mock file identity logic to avoid accessing disk
    def fake_get_identity(path):
        return _identity(str(path), f"sha_{Path(path).name}")
    
    monkeypatch.setattr("backend.sdxl_assembly.request_builder.get_file_identity", fake_get_identity)
    
    # Mock config paths
    import modules.config as config
    monkeypatch.setattr(config, "paths_checkpoints", ["/mock/checkpoints"])
    
    def fake_folder_list(name, folders):
        return f"/mock/checkpoints/{name}"
    monkeypatch.setattr("backend.sdxl_assembly.request_builder.get_file_from_folder_list", fake_folder_list)
    
    # Mock taxonomy resolution
    from modules.model_taxonomy import ARCHITECTURE_SDXL
    taxonomy = SimpleNamespace(architecture=ARCHITECTURE_SDXL)
    monkeypatch.setattr(config, "resolve_model_taxonomy", lambda path: taxonomy)

    # Mock os.path.exists and Path.exists for mock checkpoint paths
    old_exists = os.path.exists
    def fake_exists(path):
        if "mock" in str(path) or "checkpoint" in str(path):
            return True
        try:
            return old_exists(path)
        except Exception:
            return False
    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: "mock" in str(self) or "checkpoint" in str(self))

def test_normalization():
    assert _normalize_sdxl_assembly_posture_value("streaming") == "streaming"
    assert _normalize_sdxl_assembly_posture_value("auto") == "auto"
    assert _normalize_sdxl_assembly_posture_value("invalid") == "auto"
    assert _normalize_sdxl_assembly_posture_value(None) == "auto"

def test_determine_eligibility_and_vram_resolution(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    
    # 1. Default posture mapping is auto, resolved by VRAM.
    assert task.sdxl_assembly_posture == "auto"
    
    # 2. Test auto posture VRAM resolution (under 8GB)
    task.sdxl_assembly_posture = "auto"
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 4096.0)
    
    eligible, reason = determine_eligibility(task)
    assert eligible, f"Failed: {reason}"
    
    # 3. Test auto posture VRAM resolution (8GB or higher)
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 8192.0)
    eligible, reason = determine_eligibility(task)
    assert eligible, f"Failed: {reason}"

def test_posture_tuple_mapping(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task_dict = {"task_seed": 12345}
    
    # Check streaming
    task.sdxl_assembly_posture = "streaming"
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    assert req.unet_posture == UNetPostureKind.STREAMING
    assert req.clip_posture == TextEncoderPostureKind.CPU_PINNED
    assert req.vae_posture == VAEPostureKind.TRANSIENT
    assert req.lora_posture == LoraPatchPostureKind.STREAMING
    
    # Check auto with >= 8GB VRAM (maps to resident_unet_cpu_text)
    task.sdxl_assembly_posture = "auto"
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 16384.0)
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    assert req.unet_posture == UNetPostureKind.RESIDENT
    assert req.clip_posture == TextEncoderPostureKind.CPU_PINNED
    assert req.vae_posture == VAEPostureKind.TRANSIENT
    assert req.lora_posture == LoraPatchPostureKind.RESIDENT
    
    # Check auto with < 8GB VRAM (maps to streaming)
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 6144.0)
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    assert req.unet_posture == UNetPostureKind.STREAMING

def test_unsupported_posture_fails_closed():
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "invalid_posture"
    
    eligible, reason = determine_eligibility(task)
    assert not eligible
    assert "Unsupported SDXL assembly composition" in reason

def test_legacy_policy_does_not_override(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "streaming"
    task_dict = {"task_seed": 12345}
    
    # Force legacy execution policy to resident
    task.sdxl_execution_policy = SimpleNamespace(execution_mode="resident")
    
    eligible, reason = determine_eligibility(task)
    assert eligible, f"Failed: {reason}"
    
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    # The legacy policy's resident mode must not override the explicit streaming choice
    assert req.unet_posture == UNetPostureKind.STREAMING

def test_telemetry_envelope_success(monkeypatch, capsys):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "streaming"
    task_dict = {"task_seed": 12345}
    
    # Mock assembly selection and execution
    mock_assembly = MagicMock()
    mock_result = MagicMock()
    mock_result.output_image = MagicMock()
    mock_result.output_image.shape = (512, 512, 3)
    mock_assembly.execute.return_value = mock_result
    
    monkeypatch.setattr("backend.sdxl_assembly.director.SDXLAssemblyDirector.select_assembly", lambda r: mock_assembly)
    
    run_sdxl_assembly_task(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    
    captured = capsys.readouterr()
    assert "[SDXL RUN BEGIN]" in captured.out
    assert "[SDXL RUN END]" in captured.out
    assert "SUCCESS" in captured.out
    assert "Correlation ID:" in captured.out

def test_telemetry_envelope_failure(monkeypatch, capsys):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "streaming"
    task_dict = {"task_seed": 12345}
    
    # Mock failure during execution
    mock_assembly = MagicMock()
    mock_assembly.execute.side_effect = RuntimeError("Mock error execution failure")
    
    monkeypatch.setattr("backend.sdxl_assembly.director.SDXLAssemblyDirector.select_assembly", lambda r: mock_assembly)
    
    with pytest.raises(RuntimeError, match="Mock error execution failure"):
        run_sdxl_assembly_task(
            task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
        )
        
    captured = capsys.readouterr()
    assert "[SDXL RUN BEGIN]" in captured.out
    assert "[SDXL RUN END]" in captured.out
    assert "FAILURE" in captured.out
    assert "Error: Mock error execution failure" in captured.out

def test_telemetry_envelope_interrupt(monkeypatch, capsys):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "streaming"
    task_dict = {"task_seed": 12345}
    
    # Mock interrupt
    from backend.resources import InterruptProcessingException
    mock_assembly = MagicMock()
    mock_assembly.execute.side_effect = InterruptProcessingException()
    
    monkeypatch.setattr("backend.sdxl_assembly.director.SDXLAssemblyDirector.select_assembly", lambda r: mock_assembly)
    
    with pytest.raises(InterruptProcessingException):
        run_sdxl_assembly_task(
            task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
        )
        
    captured = capsys.readouterr()
    assert "[SDXL RUN BEGIN]" in captured.out
    assert "[SDXL RUN END]" in captured.out
    assert "INTERRUPT" in captured.out

def test_lifecycle_host_pinned_cache_flush():
    with patch("backend.host_cache.flush_pinned_host_cache") as mock_flush:
        release_domains(
            [LifecycleDomain.MODEL_PROMPT],
            reason="test_release",
        )
        assert mock_flush.called
