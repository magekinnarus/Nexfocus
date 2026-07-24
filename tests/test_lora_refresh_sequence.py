import types
import sys

import torch

from backend import lora as backend_lora
from backend import lora_artifacts


class _FakeInnerModel:
    def __init__(self, state_dict, unet_config=None):
        self._state_dict = state_dict
        self.model_config = types.SimpleNamespace(
            unet_config=unet_config
            or {
                "num_res_blocks": [0],
                "channel_mult": [1],
                "transformer_depth": [0],
                "transformer_depth_output": [0],
                "transformer_depth_middle": 0,
            }
        )

    def state_dict(self):
        return self._state_dict


class _FakeWrapper:
    def __init__(self, name, inner_model, applied, events):
        self.name = name
        self.model = inner_model
        self.cond_stage_model = inner_model
        self._applied = applied
        self._events = events

    def clone(self):
        return _FakeWrapper(self.name, self.model, self._applied, self._events)

    def add_patches(self, patches, weight):
        self._events.append((f"apply:{self.name}", tuple(patches.keys()), weight))
        self._applied.append((self.name, tuple(patches.keys()), weight))
        return list(patches.keys())


class _TrackingBlob:
    def __init__(self, path, released):
        self.path = path
        self._released = released

    def __del__(self):
        self._released.append(self.path)


def _compact_lora_state(prefix, out_dim=4, in_dim=8, rank=2):
    return {
        f"{prefix}.lora_up.weight": torch.ones(out_dim, rank),
        f"{prefix}.lora_down.weight": torch.ones(rank, in_dim),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }


def test_refresh_loras_builds_artifact_registry_before_application_and_uses_canonical_signature(monkeypatch, tmp_path):
    lora_a = tmp_path / "lora_a.safetensors"
    lora_b = tmp_path / "lora_b.safetensors"
    lora_a.write_bytes(b"a")
    lora_b.write_bytes(b"b")

    alias_map = {
        "alias-a-1": str(lora_a),
        "alias-b-1": str(lora_b),
        "alias-a-2": str(lora_a),
        "alias-b-2": str(lora_b),
    }

    released = []
    load_order = []
    applied = []
    events = []

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__path__ = []
    fake_transformers.__getattr__ = lambda _name: type("DummyTransformersObject", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(sys, "argv", ["pytest"], raising=False)

    import modules.core as core

    unet_inner = _FakeInnerModel({"diffusion_model.block.weight": torch.zeros(1)})
    clip_inner = _FakeInnerModel(
        {"clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.zeros(1)}
    )
    unet = _FakeWrapper("unet", unet_inner, applied, events)
    clip = _FakeWrapper("clip", clip_inner, applied, events)
    model = core.StableDiffusionModel(unet=unet, clip=clip, filename="base_model.safetensors")

    def fake_lookup(filename, *_args, **_kwargs):
        return alias_map[filename]

    def fake_load_torch_file(path, device=None):
        load_order.append(path)
        if path == str(lora_b):
            assert released and released[-1] == str(lora_a)
        return _TrackingBlob(path, released)

    def fake_load_lora(lora_sd, to_load, log_missing=False):
        assert to_load is model.lora_key_map_unet or to_load is model.lora_key_map_clip
        if to_load is model.lora_key_map_clip:
            return {
                "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": ("diff", (torch.ones(1),))
            }
        return {"diffusion_model.block.weight": ("diff", (torch.ones(1),))}

    real_normalize = lora_artifacts.normalize_loaded_lora_artifact

    def fake_normalize_loaded_lora_artifact(*, source_path, loaded_patches, default_scale=1.0, source_family="lora", source_hash=None, artifact_metadata=None):
        events.append(("normalize", source_path))
        artifact = real_normalize(
            source_path=source_path,
            loaded_patches=loaded_patches,
            default_scale=default_scale,
            source_family=source_family,
            source_hash=source_hash,
            artifact_metadata=artifact_metadata,
        )
        events.append(("artifact", artifact.source_path, artifact.default_scale))
        return artifact

    monkeypatch.setattr(core, "get_file_from_folder_list", fake_lookup)
    monkeypatch.setattr(core.backend_utils, "load_torch_file", fake_load_torch_file)
    monkeypatch.setattr(core.lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(core.lora_artifacts, "normalize_loaded_lora_artifact", fake_normalize_loaded_lora_artifact)

    model.refresh_loras([("alias-a-1", 0.5), ("alias-b-1", 1.25)])

    assert load_order == [str(lora_a), str(lora_b)]
    assert released == [str(lora_a), str(lora_b)]
    assert [event[0] for event in events[:4]] == ["normalize", "artifact", "normalize", "artifact"]
    assert [event[0] for event in events[4:]] == ["apply:unet", "apply:unet", "apply:clip", "apply:clip"]
    assert len(model.lora_artifact_registry) == 1
    stack_artifact = model.lora_artifact_registry[0]
    assert tuple(stack_artifact.artifact_metadata["stack_component_paths"]) == (str(lora_a), str(lora_b))
    assert tuple(stack_artifact.artifact_metadata["stack_component_scales"]) == (0.5, 1.25)
    assert stack_artifact.artifact_metadata["stack_component_count"] == 2
    assert applied == [
        ("unet", ("diffusion_model.block.weight",), 0.5),
        ("unet", ("diffusion_model.block.weight",), 1.25),
        ("clip", ("clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",), 0.5),
        ("clip", ("clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",), 1.25),
    ]

    model.refresh_loras([("alias-a-2", 0.5), ("alias-b-2", 1.25)])

    assert load_order == [str(lora_a), str(lora_b)]
    assert released == [str(lora_a), str(lora_b)]
    assert [event[0] for event in events[:4]] == ["normalize", "artifact", "normalize", "artifact"]
    assert len(applied) == 4
    assert len(model.lora_artifact_registry) == 1
    assert model.lora_artifact_registry[0].artifact_id == stack_artifact.artifact_id

    model.refresh_loras([("alias-a-2", 0.5), ("alias-b-2", 0.75)])

    assert load_order == [str(lora_a), str(lora_b), str(lora_a), str(lora_b)]
    assert released == [str(lora_a), str(lora_b), str(lora_a), str(lora_b)]
    assert len(model.lora_artifact_registry) == 1
    assert model.lora_artifact_registry[0].artifact_id != stack_artifact.artifact_id
    assert tuple(model.lora_artifact_registry[0].artifact_metadata["stack_component_scales"]) == (0.5, 0.75)
    assert applied[-4:] == [
        ("unet", ("diffusion_model.block.weight",), 0.5),
        ("unet", ("diffusion_model.block.weight",), 0.75),
        ("clip", ("clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",), 0.5),
        ("clip", ("clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",), 0.75),
    ]

    model.refresh_loras([("alias-a-2", 0.5)])

    assert load_order == [str(lora_a), str(lora_b), str(lora_a), str(lora_b), str(lora_a)]
    assert released == [str(lora_a), str(lora_b), str(lora_a), str(lora_b), str(lora_a)]
    assert len(model.lora_artifact_registry) == 1
    assert model.lora_artifact_registry[0].source_path == str(lora_a)
    assert applied[-2:] == [
        ("unet", ("diffusion_model.block.weight",), 0.5),
        ("clip", ("clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",), 0.5),
    ]


def test_refresh_loras_preserves_real_compact_payloads_through_artifact_bridge(monkeypatch, tmp_path):
    class _PayloadRecordingWrapper(_FakeWrapper):
        def __init__(self, name, inner_model, applied, events, payloads):
            super().__init__(name, inner_model, applied, events)
            self._payloads = payloads

        def clone(self):
            return _PayloadRecordingWrapper(self.name, self.model, self._applied, self._events, self._payloads)

        def add_patches(self, patches, weight):
            self._payloads.append((self.name, dict(patches), weight))
            return super().add_patches(patches, weight)

    lora_path = tmp_path / "compact_bridge.safetensors"
    lora_path.write_bytes(b"bridge")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__path__ = []
    fake_transformers.__getattr__ = lambda _name: type("DummyTransformersObject", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(sys, "argv", ["pytest"], raising=False)

    import modules.core as core

    applied = []
    events = []
    payloads = []
    bridge_calls = []

    unet_inner = _FakeInnerModel({"diffusion_model.block.weight": torch.zeros(4, 8)})
    clip_target_key = "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    clip_inner = _FakeInnerModel({clip_target_key: torch.zeros(4, 8)})

    unet = _PayloadRecordingWrapper("unet", unet_inner, applied, events, payloads)
    clip = _PayloadRecordingWrapper("clip", clip_inner, applied, events, payloads)
    model = core.StableDiffusionModel(unet=unet, clip=clip, filename="base_model.safetensors")

    unet_prefix = next(
        key for key, value in model.lora_key_map_unet.items()
        if value == "diffusion_model.block.weight" and key.startswith("lora_unet_")
    )
    clip_prefix = next(
        key for key, value in model.lora_key_map_clip.items()
        if value == clip_target_key and key.startswith("text_encoders.")
    )

    lora_sd = {}
    lora_sd.update(_compact_lora_state(unet_prefix))
    lora_sd.update(_compact_lora_state(clip_prefix))

    monkeypatch.setattr(core, "get_file_from_folder_list", lambda *_args, **_kwargs: str(lora_path))
    monkeypatch.setattr(core.backend_utils, "load_torch_file", lambda *_args, **_kwargs: dict(lora_sd))

    real_build_application_patch_dict = core.lora_artifacts.build_application_patch_dict

    def fake_build_application_patch_dict(artifact, key_map, *, target_family=None):
        bridge_calls.append((artifact.source_path, target_family, len(key_map)))
        return real_build_application_patch_dict(artifact, key_map, target_family=target_family)

    monkeypatch.setattr(core.lora_artifacts, "build_application_patch_dict", fake_build_application_patch_dict)

    model.refresh_loras([("alias-compact", 0.75)])

    assert len(model.lora_artifact_registry) == 1
    artifact = model.lora_artifact_registry[0]
    artifact_entries = {entry.target_key: entry for entry in artifact.target_entries}

    assert artifact.default_scale == 0.75
    assert artifact_entries["diffusion_model.block.weight"].payload_family == "lora"
    assert artifact_entries[clip_target_key].payload_family == "lora"
    assert artifact_entries["diffusion_model.block.weight"].supports_compact_retention is True
    assert artifact_entries[clip_target_key].supports_compact_retention is True

    assert len(payloads) == 2
    assert payloads[0][0] == "unet"
    assert payloads[0][2] == 0.75
    assert tuple(payloads[0][1].keys()) == ("diffusion_model.block.weight",)
    assert payloads[0][1]["diffusion_model.block.weight"] is artifact_entries["diffusion_model.block.weight"].payload
    assert getattr(payloads[0][1]["diffusion_model.block.weight"], "name", None) == "lora"

    assert payloads[1][0] == "clip"
    assert payloads[1][2] == 0.75
    assert tuple(payloads[1][1].keys()) == (clip_target_key,)
    assert payloads[1][1][clip_target_key] is artifact_entries[clip_target_key].payload
    assert getattr(payloads[1][1][clip_target_key], "name", None) == "lora"
    assert bridge_calls == [
        (str(lora_path), "unet", len(model.lora_key_map_unet)),
        (str(lora_path), "clip", len(model.lora_key_map_clip)),
    ]


def test_lora_key_map_helpers_allocate_fresh_dicts():
    unet_model = _FakeInnerModel({"diffusion_model.block.weight": torch.zeros(1)})
    clip_model = _FakeInnerModel(
        {"clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.zeros(1)}
    )

    first_unet = backend_lora.model_lora_keys_unet(unet_model)
    second_unet = backend_lora.model_lora_keys_unet(unet_model)
    first_unet["sentinel"] = "value"

    first_clip = backend_lora.model_lora_keys_clip(clip_model)
    second_clip = backend_lora.model_lora_keys_clip(clip_model)
    first_clip["sentinel"] = "value"

    assert first_unet is not second_unet
    assert "sentinel" not in second_unet
    assert first_clip is not second_clip
    assert "sentinel" not in second_clip
