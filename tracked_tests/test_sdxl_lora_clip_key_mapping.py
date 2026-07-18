from pathlib import Path
from types import SimpleNamespace

import torch

from backend import lora as backend_lora
from backend.sdxl_assembly import cpu_lora_worker


class _NexClipModel:
    def state_dict(self):
        return {
            "clip_l.transformer.encoder.layers.0.self_attn.q_proj.weight": torch.empty(1),
            "clip_l.transformer.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
            "clip_g.transformer.encoder.layers.0.self_attn.q_proj.weight": torch.empty(1),
            "clip_g.transformer.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
            "clip_g.text_projection": torch.empty(1),
        }


class _LegacyClipModel:
    def state_dict(self):
        return {
            "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.empty(1),
            "clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
            "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
        }


def _spec(name, sha, *, unet_weight=1.0, clip_weight=1.0):
    return SimpleNamespace(
        enabled=True,
        unet_weight=unet_weight,
        clip_weight=clip_weight,
        file_identity=SimpleNamespace(path=Path(name), sha256=sha),
    )


def test_model_lora_keys_clip_maps_kohya_sdxl_names_to_nex_clip_layout():
    key_map = backend_lora.model_lora_keys_clip(_NexClipModel())

    assert key_map["lora_te1_text_model_encoder_layers_0_self_attn_q_proj"] == (
        "clip_l.transformer.encoder.layers.0.self_attn.q_proj.weight"
    )
    assert key_map["lora_te2_text_model_encoder_layers_0_mlp_fc1"] == (
        "clip_g.transformer.encoder.layers.0.mlp.fc1.weight"
    )
    assert key_map["lora_te2_text_projection"] == "clip_g.text_projection"


def test_model_lora_keys_clip_preserves_legacy_clip_layout_mapping():
    key_map = backend_lora.model_lora_keys_clip(_LegacyClipModel())

    assert key_map["lora_te1_text_model_encoder_layers_0_self_attn_q_proj"] == (
        "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
    )
    assert key_map["lora_te2_text_model_encoder_layers_0_mlp_fc1"] == (
        "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    )


def test_mixed_dual_and_unet_only_assets_keep_resolved_clip_patches(monkeypatch):
    dual_key = "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
    headers = {
        "dual.safetensors": {
            f"{dual_key}.alpha": torch.tensor(1.0),
            f"{dual_key}.lora_down.weight": torch.ones((1, 1)),
            f"{dual_key}.lora_up.weight": torch.ones((1, 1)),
        },
        # This adapter has a requested CLIP weight but contains only UNet tensors.
        "unet_only.safetensors": {
            "lora_unet_input_blocks_0_0.lora_down.weight": torch.ones((1, 1)),
            "lora_unet_input_blocks_0_0.lora_up.weight": torch.ones((1, 1)),
        },
    }
    monkeypatch.setattr(
        cpu_lora_worker,
        "SafeOpenHeaderOnly",
        lambda path: headers[Path(path).name],
    )
    cpu_lora_worker._PARSED_LORA_CACHE.clear()

    request = SimpleNamespace(
        lora_specs=(
            _spec("dual.safetensors", "dual_sha"),
            _spec("unet_only.safetensors", "unet_only_sha"),
        )
    )
    clip = SimpleNamespace(patcher=SimpleNamespace(model=_NexClipModel()))

    resolved = cpu_lora_worker.CpuLoraWorker(request).resolve_clip_patches(clip)

    assert len(resolved) == 1
    assert resolved[0][1] == 1.0
    assert tuple(resolved[0][0]) == (
        "clip_l.transformer.encoder.layers.0.self_attn.q_proj.weight",
    )
    cpu_lora_worker._PARSED_LORA_CACHE.clear()
