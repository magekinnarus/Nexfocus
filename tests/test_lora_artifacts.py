import types

import torch

from backend import conditioning, resources
from backend import lora as backend_lora
from backend import lora_artifacts
from backend import environment_profile
from backend import sdxl_runtime_policy


class _FakeModel:
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


def _lora_state(prefix, out_dim=4, in_dim=8, rank=2):
    return {
        f"{prefix}.lora_up.weight": torch.ones(out_dim, rank),
        f"{prefix}.lora_down.weight": torch.ones(rank, in_dim),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }


def test_artifact_records_preserve_separate_source_identity():
    target_key = "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight"
    to_load = {"lora_unet_down_attn": target_key}
    patches = backend_lora.load_lora(_lora_state("lora_unet_down_attn"), to_load, log_missing=False)

    first = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/first.safetensors",
        source_hash="hash-one",
        default_scale=0.5,
        loaded_patches=patches,
    )
    second = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/second.safetensors",
        source_hash="hash-two",
        default_scale=0.5,
        loaded_patches=patches,
    )

    assert first.source_path != second.source_path
    assert first.source_hash != second.source_hash
    assert first.artifact_id != second.artifact_id
    assert first.target_entries[0].payload is patches[target_key]
    assert second.target_entries[0].payload is patches[target_key]


def test_artifact_identity_derives_from_payload_when_source_hash_is_missing():
    target_key = "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight"
    to_load = {"lora_unet_down_attn": target_key}
    first_patches = backend_lora.load_lora(_lora_state("lora_unet_down_attn", rank=2), to_load, log_missing=False)
    second_patches = backend_lora.load_lora(_lora_state("lora_unet_down_attn", rank=3), to_load, log_missing=False)

    first = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/shared-path.safetensors",
        loaded_patches=first_patches,
    )
    second = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/shared-path.safetensors",
        loaded_patches=second_patches,
    )

    assert first.source_hash != second.source_hash
    assert first.artifact_id != second.artifact_id
    assert first.artifact_metadata["source_hash_origin"] == "payload"
    assert second.artifact_metadata["source_hash_origin"] == "payload"


def test_normalization_preserves_compact_lora_payload_from_real_loader():
    target_key = "diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight"
    to_load = {"lora_unet_up_ff": target_key}
    patches = backend_lora.load_lora(_lora_state("lora_unet_up_ff", rank=3), to_load, log_missing=False)

    artifact = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/compact.safetensors",
        source_hash="hash-compact",
        default_scale=1.25,
        loaded_patches=patches,
    )

    entry = artifact.target_entries[0]
    assert entry.target_key == target_key
    assert entry.payload is patches[target_key]
    assert entry.payload_family == "lora"
    assert entry.source_rank == 3
    assert entry.supports_compact_retention is True
    assert entry.target_group == "unet.up"
    assert entry.target_subgroup == "unet.up.ff"
    assert entry.block_tag is None


def test_build_application_patch_dict_preserves_compact_payload_identity():
    target_key = "diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight"
    to_load = {"lora_unet_up_ff": target_key}
    patches = backend_lora.load_lora(_lora_state("lora_unet_up_ff", rank=3), to_load, log_missing=False)

    artifact = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/compact.safetensors",
        source_hash="hash-compact",
        default_scale=1.25,
        loaded_patches=patches,
    )

    key_map = {target_key: target_key}
    patch_dict = lora_artifacts.build_application_patch_dict(artifact, key_map, target_family="unet")

    assert patch_dict[target_key] is artifact.target_entries[0].payload
    assert getattr(patch_dict[target_key], "name", None) == "lora"
    assert tuple(patch_dict.keys()) == (target_key,)


def test_merge_loaded_lora_artifacts_creates_single_stack_identity_and_replays_components_in_order():
    first = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/first.safetensors",
        source_hash="hash-first",
        default_scale=0.5,
        loaded_patches={"diffusion_model.block.weight": torch.ones(2, 2)},
    )
    second = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/second.safetensors",
        source_hash="hash-second",
        default_scale=1.25,
        loaded_patches={"clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.full((2, 2), 2.0)},
    )

    merged = lora_artifacts.merge_loaded_lora_artifacts(
        (first, second),
        source_path="stack::first+second",
    )

    assert merged.artifact_id != first.artifact_id
    assert merged.artifact_id != second.artifact_id
    assert merged.artifact_metadata["stack_component_count"] == 2
    assert merged.artifact_metadata["stack_component_ids"] == (first.artifact_id, second.artifact_id)
    assert merged.artifact_metadata["stack_component_scales"] == (0.5, 1.25)
    assert lora_artifacts.artifact_registry_signature((merged,)) == (merged.artifact_id,)

    class _RecordingPatcher:
        def __init__(self):
            self.calls = []

        def add_patches(self, patches, weight):
            self.calls.append((dict(patches), weight))
            return tuple(patches.keys())

    patcher = _RecordingPatcher()
    key_map = {
        "diffusion_model.block.weight": "diffusion_model.block.weight",
        "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",
    }

    loaded_keys = lora_artifacts.apply_artifact_to_patcher(patcher, merged, key_map)

    assert loaded_keys == [
        "diffusion_model.block.weight",
        "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight",
    ]
    assert patcher.calls == [
        ({"diffusion_model.block.weight": first.target_entries[0].payload}, 0.5),
        ({"clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": second.target_entries[0].payload}, 1.25),
    ]


def test_target_group_classification_is_deterministic_for_unet_and_clip():
    cases = {
        "diffusion_model.input_blocks.1.0.in_layers.2.weight": ("unet.down", "unet.down.conv", None),
        "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": (
            "unet.mid",
            "unet.mid.attn",
            "mid.block.0",
        ),
        "diffusion_model.output_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": (
            "unet.up",
            "unet.up.ff",
            None,
        ),
        "clip_l.transformer.text_model.encoder.layers.10.mlp.fc1.weight": (
            "clip",
            "clip.mlp",
            None,
        ),
        "clip_g.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight": (
            "clip",
            "clip.attn",
            None,
        ),
    }

    for key, expected in cases.items():
        group, subgroup, block_tag = expected
        assert lora_artifacts.classify_target_group(key) == group
        assert lora_artifacts.classify_target_subgroup(key) == subgroup
        assert lora_artifacts.extract_block_tag(key) == block_tag
        assert lora_artifacts.classify_target_group(key) == group


def test_diffusers_style_block_tags_use_structural_indices():
    assert lora_artifacts.extract_block_tag("down_blocks.1.attentions.0.to_q.weight") == "down.block.1"
    assert lora_artifacts.extract_block_tag("mid_block.attentions.0.to_q.weight") == "mid.block.0"
    assert lora_artifacts.extract_block_tag("up_blocks.2.attentions.0.to_q.weight") == "up.block.2"


def test_clip_target_family_maps_from_real_loader_output():
    target_key = "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    to_load = {"lora_te2_text_model_encoder_layers_0_mlp_fc1": target_key}
    patches = backend_lora.load_lora(
        _lora_state("lora_te2_text_model_encoder_layers_0_mlp_fc1"),
        to_load,
        log_missing=False,
    )

    artifact = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/clip.safetensors",
        loaded_patches=patches,
        source_hash="clip-hash",
    )

    entry = artifact.target_entries[0]
    assert entry.target_family == "clip"
    assert entry.target_group == "clip"
    assert entry.target_subgroup == "clip.mlp"
    assert entry.payload_family == "lora"


def test_dense_payloads_are_identified_as_non_compact():
    target_key = "diffusion_model.middle_block.1.proj_in.weight"
    payload = ("diff", (torch.ones(4, 4),))

    artifact = lora_artifacts.normalize_loaded_lora_artifact(
        source_path="D:/AI/Imagine/models/loras/SDXL/diff.safetensors",
        loaded_patches={target_key: payload},
        source_hash="diff-hash",
    )

    entry = artifact.target_entries[0]
    assert entry.payload is payload
    assert entry.payload_family == "diff"
    assert entry.supports_compact_retention is False
    assert entry.target_group == "unet.mid"


def test_lora_key_map_helpers_allocate_fresh_dicts():
    unet_model = _FakeModel({"diffusion_model.block.weight": torch.zeros(1)})
    clip_model = _FakeModel(
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


def test_sdxl_residency_class_normalization():
    assert resources.normalize_sdxl_residency_class() == resources.SDXL_RESIDENCY_CLASS_FULL
    assert resources.normalize_sdxl_residency_class("full_resident") == resources.SDXL_RESIDENCY_CLASS_FULL
    assert resources.normalize_sdxl_residency_class("unified_streaming") == resources.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING


def test_sdxl_text_conditioning_fingerprint_includes_residency_class_and_lora_identity():
    artifacts = (
        lora_artifacts.AdapterArtifact(
            artifact_id="artifact-a",
            source_path="a.safetensors",
            source_family="lora",
            source_hash="hash-a",
            default_scale=1.0,
            target_entries=(),
        ),
        lora_artifacts.AdapterArtifact(
            artifact_id="artifact-b",
            source_path="b.safetensors",
            source_family="lora",
            source_hash="hash-b",
            default_scale=0.75,
            target_entries=(),
        ),
    )

    left = conditioning.build_sdxl_text_conditioning_fingerprint(
        prompt="hello",
        negative_prompt="bad",
        model_identity="model-x",
        text_encoder_identity="clip-y",
        clip_patch_uuid="clip-patch-1",
        clip_layer_idx=-2,
        lora_artifacts_state=artifacts,
        route_family_reconciliation_signature="route-gen-1",
        residency_class=resources.SDXL_RESIDENCY_CLASS_FULL,
        route_family="txt2img",
    )
    right = conditioning.build_sdxl_text_conditioning_fingerprint(
        prompt="hello",
        negative_prompt="bad",
        model_identity="model-x",
        text_encoder_identity="clip-y",
        clip_patch_uuid="clip-patch-1",
        clip_layer_idx=-2,
        lora_artifacts_state=artifacts,
        route_family_reconciliation_signature="route-gen-1",
        residency_class=resources.SDXL_RESIDENCY_CLASS_FULL,
        route_family="txt2img",
    )
    streaming = conditioning.build_sdxl_text_conditioning_fingerprint(
        prompt="hello",
        negative_prompt="bad",
        model_identity="model-x",
        text_encoder_identity="clip-y",
        clip_patch_uuid="clip-patch-1",
        clip_layer_idx=-2,
        lora_artifacts_state=artifacts,
        route_family_reconciliation_signature="route-gen-1",
        residency_class=resources.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        route_family="txt2img",
    )

    assert left == right
    assert left != streaming
    assert left.digest() != streaming.digest()
    assert ("route_family", "txt2img") in left.components
    assert ("lora_signature", ("artifact-a", "artifact-b")) in left.components
    assert lora_artifacts.artifact_registry_signature(artifacts) == ("artifact-a", "artifact-b")


def test_sdxl_prepared_payload_fingerprint_freezes_nested_payloads():
    fingerprint = conditioning.build_sdxl_prepared_payload_fingerprint(
        "vae_encode",
        residency_class=resources.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        model_identity="model-x",
        route_family_reconciliation_signature="route-gen-2",
        prepared_artifact_signature={"image": "hash-image", "mask": ["hash-mask-a", "hash-mask-b"]},
        canvas={"width": 1024, "height": 1024},
    )

    assert fingerprint.stage_name == "vae_encode"
    assert fingerprint.residency_class == resources.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING
    assert ("prepared_artifact_signature", (("image", "hash-image"), ("mask", ("hash-mask-a", "hash-mask-b")))) in fingerprint.components


def test_sdxl_execution_policy_keeps_standard_checkpoint_path():
    low_vram_profile = environment_profile.EnvironmentProfile(
        name=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        display_name="Local Low VRAM",
        source="test",
        total_ram_mb=16384.0,
        total_vram_mb=6144.0,
        is_colab=False,
    )
    colab_free_profile = environment_profile.EnvironmentProfile(
        name=environment_profile.PROFILE_COLAB_FREE,
        display_name="Colab Free",
        source="test",
        total_ram_mb=12288.0,
        total_vram_mb=15360.0,
        is_colab=True,
    )

    low_vram_standard = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="juggernautXL_v9.safetensors",
        profile=low_vram_profile,
    )
    colab_free_standard = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="juggernautXL_v9.safetensors",
        profile=colab_free_profile,
    )

    assert low_vram_standard.execution_family == sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD
    assert low_vram_standard.runtime_family == "unified_sdxl"
    assert low_vram_standard.notes

    assert colab_free_standard.execution_family == sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD
    assert colab_free_standard.residency_class == sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL
    assert colab_free_standard.clip_residency_mode in {
        sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        sdxl_runtime_policy.CLIP_RESIDENCY_GPU_RESIDENT,
    }
    assert colab_free_standard.vae_encode_mode in {
        sdxl_runtime_policy.VAE_ENCODE_CPU_DEFAULT,
        sdxl_runtime_policy.VAE_ENCODE_GPU_PREFERRED,
        sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
        sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
    }
    assert isinstance(colab_free_standard.keep_clip_loaded, bool)


def test_sdxl_execution_policy_treats_colab_pro_and_12gb_plus_as_gpu_preferred():
    colab_pro_profile = environment_profile.EnvironmentProfile(
        name=environment_profile.PROFILE_COLAB_PRO,
        display_name="Colab Pro",
        source="test",
        total_ram_mb=24576.0,
        total_vram_mb=24576.0,
        is_colab=True,
    )
    local_high_vram_profile = environment_profile.EnvironmentProfile(
        name=environment_profile.PROFILE_LOCAL_NORMAL,
        display_name="Local Normal",
        source="test",
        total_ram_mb=32768.0,
        total_vram_mb=12288.0,
        is_colab=False,
    )

    colab_pro_standard = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="juggernautXL_v9.safetensors",
        profile=colab_pro_profile,
    )
    local_high_vram_standard = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="juggernautXL_v9.safetensors",
        profile=local_high_vram_profile,
    )
    local_odd_streaming_profile = environment_profile.EnvironmentProfile(
        name=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        display_name="Local Odd Streaming",
        source="test",
        total_ram_mb=16384.0,
        total_vram_mb=7168.0,
        is_colab=False,
    )
    local_odd_streaming_standard = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="juggernautXL_v9.safetensors",
        profile=local_odd_streaming_profile,
    )

    for policy in (colab_pro_standard, local_high_vram_standard):
        assert policy.execution_family == sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD
        assert policy.residency_class == sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL
        assert policy.clip_residency_mode in {
            sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
            sdxl_runtime_policy.CLIP_RESIDENCY_GPU_RESIDENT,
        }
        assert policy.vae_encode_mode in {
            sdxl_runtime_policy.VAE_ENCODE_CPU_DEFAULT,
            sdxl_runtime_policy.VAE_ENCODE_GPU_PREFERRED,
            sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
            sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
        }
        assert isinstance(policy.keep_clip_loaded, bool)

    assert local_odd_streaming_standard.execution_family == sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD
    assert local_odd_streaming_standard.residency_class == sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING
    assert local_odd_streaming_standard.vae_encode_mode == sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU
