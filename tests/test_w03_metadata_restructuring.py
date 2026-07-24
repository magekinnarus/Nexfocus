"""
Automated unit & regression test suite for P5-M02-W03 Metadata Restructuring.
Verifies v2 schema serialization, v1 backward-compatibility shim, prompt routing,
category-grouped preview formatting, clipboard JSON payload, and default settings.
"""

import json
import pytest
import re
from types import SimpleNamespace
import urllib.parse
from unittest.mock import MagicMock

import modules.config as config
import modules.meta_parser as meta_parser
import modules.parameter_registry as parameter_registry
import modules.ui_components.metadata_ui as metadata_ui
import modules.ui_components.metadata_preview as metadata_preview
from modules.flags import MetadataScheme
from modules.pipeline.output import _resolve_workflow_identity, save_and_log
from modules.task_state import TaskState


@pytest.fixture(autouse=True)
def isolate_w03_output_paths(tmp_path, monkeypatch):
    import args_manager
    import modules.private_logger as private_logger

    monkeypatch.setattr(config, "path_outputs", str(tmp_path))
    monkeypatch.setattr(private_logger.modules.config, "path_outputs", str(tmp_path))
    monkeypatch.setattr(args_manager.args, "disable_image_log", False, raising=False)


def test_v2_schema_txt2img_serialization(tmp_path):
    """Test 1: Every saved image carries v2 metadata with metadata_version=2 and workflow field."""
    task_state = TaskState(
        prompt="A majestic lion on a cliff",
        negative_prompt="blurry, low quality",
        base_model_name="sd_xl_base_1.0.safetensors",
        vae_name="sdxl_vae.safetensors",
        save_metadata_to_images=True,
        metadata_scheme=MetadataScheme.FOOOCUS_NEX,
        steps=30,
        cfg_scale=7.0,
        seed=12345,
    )
    task_dict = {
        'log_positive_prompt': "A majestic lion on a cliff",
        'log_negative_prompt': "blurry, low quality",
        'positive': ["A majestic lion on a cliff"],
        'negative': ["blurry, low quality"],
        'styles': ["Fooocus Sharp"],
        'task_seed': 12345,
    }
    dummy_img = (255 * pytest.importorskip("numpy").zeros((64, 64, 3), dtype="uint8"))

    # Mock path outputs to temp
    config.path_outputs = str(tmp_path)
    img_paths = save_and_log(
        task_state, 1024, 1024, [dummy_img], task_dict, False, [("lora_a", 0.8)]
    )
    assert len(img_paths) == 1
    file_path = img_paths[0]

    parameters, scheme = meta_parser.read_info_from_image(file_path)
    assert parameters is not None
    assert parameters.get("metadata_version") == 2
    assert parameters.get("workflow") == "txt2img"
    assert parameters.get("prompt") == "A majestic lion on a cliff"
    assert parameters.get("seed") == 12345
    assert parameters.get("base_model") == "sd_xl_base_1.0"
    assert parameters.get("version") == "Nexfocus V1.0.0"
    assert "base_model_hash" in parameters
    assert "sharpness" in parameters  # Stored in record as hidden field


def test_flux_fill_inpaint_schema_excludes_sdxl_fields(tmp_path):
    """Test 3: Flux Fill records contain no SDXL fields."""
    task_state = TaskState(
        save_metadata_to_images=True,
        metadata_scheme=MetadataScheme.FOOOCUS_NEX,
        steps=25,
        sampler_name="euler",
        scheduler_name="simple",
        seed=999,
        flux_fill_t5_path="t5xxl_fp16.safetensors",
        flux_fill_ae_path="ae.safetensors",
    )
    task_dict = {
        'log_positive_prompt': "Repair the background wall",
        'description': "Flux Fill Inpaint",
        'task_seed': 999,
    }
    dummy_img = (255 * pytest.importorskip("numpy").zeros((64, 64, 3), dtype="uint8"))
    config.path_outputs = str(tmp_path)

    img_paths = save_and_log(
        task_state, 1024, 1024, [dummy_img], task_dict, False, [], workflow="flux_fill_inpaint"
    )
    parameters, _ = meta_parser.read_info_from_image(img_paths[0])
    assert parameters is not None
    assert parameters.get("metadata_version") == 2
    assert parameters.get("workflow") == "flux_fill_inpaint"

    # Absent SDXL fields
    assert "cfg_scale" not in parameters
    assert "clip_skip" not in parameters
    assert "vae" not in parameters
    assert "negative_prompt" not in parameters
    assert "styles" not in parameters
    assert "base_model" not in parameters
    assert "loras" not in parameters
    assert parameters.get("t5") == "t5xxl_fp16"
    assert parameters.get("ae") == "ae"


def test_non_generative_workflows_carry_no_deployable_fields(tmp_path):
    """Test 4: Non-generative workflows carry no deployable fields."""
    task_state = TaskState(
        save_metadata_to_images=True,
        metadata_scheme=MetadataScheme.FOOOCUS_NEX,
    )
    task_dict = {'task_seed': -1}
    dummy_img = (255 * pytest.importorskip("numpy").zeros((64, 64, 3), dtype="uint8"))
    config.path_outputs = str(tmp_path)

    img_paths = save_and_log(
        task_state, 512, 512, [dummy_img], task_dict, False, [], workflow="remove_mat"
    )
    parameters, _ = meta_parser.read_info_from_image(img_paths[0])
    assert parameters is not None
    assert parameters.get("workflow") == "remove_mat"
    assert parameters.get("resolution") == "512x512"
    assert "prompt" not in parameters
    assert "base_model" not in parameters
    assert "seed" not in parameters


def test_v1_backward_compatibility_shim():
    """Test 5: Old v1 metadata is still readable and produces v2 record on import."""
    v1_record = {
        "Prompt": "A vintage car on route 66",
        "Negative Prompt": "ugly, blurry",
        "Styles": "['Fooocus Sharp']",
        "Base Model": "sd_xl_base_1.0.safetensors",
        "Steps": "30",
        "Guidance Scale": "7.5",
        "Seed": "54321",
        "Resolution": "(1024, 1024)",
        "Sampler": "dpmpp_2m",
        "Scheduler": "karras",
    }

    converted = meta_parser.convert_v1_to_v2_metadata(v1_record)
    assert converted.get("metadata_version") == 2
    assert converted.get("workflow") == "txt2img"
    assert converted.get("prompt") == "A vintage car on route 66"
    assert converted.get("seed") == 54321
    assert converted.get("cfg_scale") == 7.5
    assert converted.get("styles") == ["Fooocus Sharp"]


def test_prompt_routing_on_import():
    """Test 2: Metadata import correctly routes prompts to their owning tab."""
    # txt2img record
    txt2img_meta = {
        "metadata_version": 2,
        "workflow": "txt2img",
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "fog, clouds",
        "styles": [],
        "seed": 101,
    }
    res_txt = metadata_ui.load_parameter_button_click(txt2img_meta, is_generating=False)
    assert res_txt[metadata_ui.METADATA_OUTPUT_INDEX['prompt']] == "A beautiful sunset over mountains"

    # inpaint_sdxl record
    inpaint_meta = {
        "metadata_version": 2,
        "workflow": "inpaint_sdxl",
        "prompt": "Full scene background",
        "inpaint_prompt": "Golden hair piece",
        "inpaint_route": "sdxl",
    }
    res_inp = metadata_ui.load_parameter_button_click(inpaint_meta, is_generating=False)
    assert res_inp[metadata_ui.METADATA_OUTPUT_INDEX['prompt']] == "Full scene background"
    assert res_inp[metadata_ui.METADATA_OUTPUT_INDEX['inpaint_prompt']] == "Golden hair piece"
    assert res_inp[metadata_ui.METADATA_OUTPUT_INDEX['inpaint_route']] == "sdxl"

    # Route-owned patch engine fields must apply only to their owning tab.
    inpaint_meta['inpaint_engine'] = 'v2.6'
    res_inp = metadata_ui.load_parameter_button_click(inpaint_meta, is_generating=False)
    assert res_inp[metadata_ui.METADATA_OUTPUT_INDEX['inpaint_engine']] == 'v2.6'
    assert res_inp[metadata_ui.METADATA_OUTPUT_INDEX['outpaint_engine']] == {'__type__': 'update'}

    outpaint_meta = {
        "metadata_version": 2,
        "workflow": "outpaint_sdxl",
        "prompt": "Full scene background",
        "outpaint_prompt": "Extend the mountains",
        "outpaint_engine": "v2.6",
    }
    res_out = metadata_ui.load_parameter_button_click(outpaint_meta, is_generating=False)
    assert res_out[metadata_ui.METADATA_OUTPUT_INDEX['outpaint_engine']] == 'v2.6'
    assert res_out[metadata_ui.METADATA_OUTPUT_INDEX['inpaint_engine']] == {'__type__': 'update'}


@pytest.mark.parametrize(
    ('workflow', 'engine_field', 'engine_attr', 'prompt_attr'),
    [
        ('inpaint_sdxl', 'inpaint_engine', 'inpaint_engine', 'inpaint_additional_prompt'),
        ('outpaint_sdxl', 'outpaint_engine', 'outpaint_engine', 'outpaint_additional_prompt'),
    ],
)
def test_sdxl_image_input_engine_is_serialized(tmp_path, workflow, engine_field, engine_attr, prompt_attr):
    task_state = TaskState(
        prompt='Main scene',
        base_model_name='sd_xl_base_1.0.safetensors',
        vae_name='sdxl_vae.safetensors',
        save_metadata_to_images=True,
        metadata_scheme=MetadataScheme.FOOOCUS_NEX,
        steps=20,
        sampler_name='dpmpp_2m',
        scheduler_name='beta',
        cfg_scale=7.0,
        seed=123,
        inpaint_engine='None',
        outpaint_engine='None',
    )
    setattr(task_state, engine_attr, 'v2.6')
    setattr(task_state, prompt_attr, 'Repair or extend the image')
    task_dict = {'task_seed': 123, 'styles': [], 'log_negative_prompt': ''}
    dummy_img = (255 * pytest.importorskip('numpy').zeros((16, 16, 3), dtype='uint8'))

    img_path = save_and_log(
        task_state, 512, 512, [dummy_img], task_dict, False, [], workflow=workflow
    )[0]
    parameters, _ = meta_parser.read_info_from_image(img_path)

    assert parameters[engine_field] == 'v2.6'
    assert parameters['workflow'] == workflow
    assert parameters[prompt_attr.replace('_additional_prompt', '_prompt')] == 'Repair or extend the image'


def test_category_grouped_preview_formatting():
    """Test 6 & 7: Category-grouped preview showing deployable fields, display-only, and omitting hidden."""
    v2_record = {
        "metadata_version": 2,
        "workflow": "txt2img",
        "version": "Nex V1.0.0",
        "prompt": "Cyberpunk city alley",
        "base_model": "sd_xl_base_1.0",
        "seed": 777,
        "vae": "sdxl_vae",
        "sharpness": 2.0,
        "adm_guidance": (1.5, 0.8, 0.3),
        "adaptive_cfg": 7.0,
    }

    rendered = metadata_preview.format_metadata_preview(v2_record)
    assert "Workflow: txt2img (v2)" in rendered
    assert "Deployable Parameters:" in rendered
    assert "prompt: Cyberpunk city alley" in rendered
    assert "Display-Only Reference:" in rendered
    assert "vae: sdxl_vae" in rendered

    # Hidden fields MUST NOT appear in preview
    assert "sharpness" not in rendered
    assert "adm_guidance" not in rendered
    assert "adaptive_cfg" not in rendered


def test_save_metadata_to_images_defaults_to_true():
    """Test 9: save_metadata_to_images defaults to True."""
    registry_def = next(p for p in parameter_registry.PARAM_REGISTRY if p.name == "save_metadata_to_images")
    assert registry_def.default is True
    assert config.default_save_metadata_to_images is True


@pytest.mark.parametrize("output_format", ["png", "jpeg", "webp"])
def test_v2_metadata_round_trips_all_supported_output_formats(tmp_path, monkeypatch, output_format):
    monkeypatch.setattr(config, "path_outputs", str(tmp_path))
    task_state = TaskState(
        prompt="format round trip",
        base_model_name="sd_xl_base_1.0.safetensors",
        save_metadata_to_images=True,
        metadata_scheme=MetadataScheme.FOOOCUS_NEX,
        output_format=output_format,
        seed=42,
    )
    task_dict = {
        "log_positive_prompt": "format round trip",
        "log_negative_prompt": "",
        "positive": ["format round trip"],
        "negative": [],
        "styles": [],
        "task_seed": 42,
    }
    dummy_img = pytest.importorskip("numpy").zeros((16, 16, 3), dtype="uint8")

    path = save_and_log(task_state, 16, 16, [dummy_img], task_dict, False, [])[0]
    parameters, scheme = meta_parser.read_info_from_image(path)

    assert parameters["metadata_version"] == 2
    assert parameters["workflow"] == "txt2img"
    assert parameters["version"] == "Nexfocus V1.0.0"
    assert scheme == MetadataScheme.FOOOCUS_NEX


def test_workflow_identity_uses_current_taxonomy_and_frozen_plan():
    assert _resolve_workflow_identity(
        SimpleNamespace(workflow_plan=None, uov_method="Upscale", current_tab="uov", goals=[]),
        {},
    ) == "upscale_gan"
    assert _resolve_workflow_identity(
        SimpleNamespace(workflow_plan=None, uov_method="Super-Upscale", current_tab="uov", goals=[]),
        {},
    ) == "super_upscale"
    assert _resolve_workflow_identity(
        SimpleNamespace(workflow_plan=SimpleNamespace(route_id="outpaint")),
        {},
    ) == "outpaint_sdxl"
    assert _resolve_workflow_identity(
        SimpleNamespace(workflow_plan=SimpleNamespace(route_id="flux_removal")),
        {},
    ) == "flux_fill_remove"


@pytest.mark.parametrize(
    ("workflow", "state_fields", "expected_field", "expected_value"),
    [
        (
            "inpaint_sdxl",
            {"prompt": "main", "inpaint_additional_prompt": "repair hair"},
            "inpaint_prompt",
            "repair hair",
        ),
        (
            "outpaint_sdxl",
            {"prompt": "main", "outpaint_additional_prompt": "extend mountain"},
            "outpaint_prompt",
            "extend mountain",
        ),
        (
            "flux_fill_remove",
            {"remove_prompt": "replace with beach"},
            "prompt_description",
            "replace with beach",
        ),
        (
            "color_enhance",
            {"upscale_prompt": "warm cinematic colors"},
            "prompt_description",
            "warm cinematic colors",
        ),
    ],
)
def test_serializer_preserves_workflow_owned_prompt(
    monkeypatch,
    workflow,
    state_fields,
    expected_field,
    expected_value,
):
    captured = {}

    def fake_log(*args, **kwargs):
        captured.update(kwargs["clipboard_metadata"])
        return "captured.png"

    monkeypatch.setattr("modules.private_logger.log", fake_log)
    task_state = TaskState(
        base_model_name="sd_xl_base_1.0.safetensors",
        save_metadata_to_images=False,
        seed=7,
        **state_fields,
    )
    task_dict = {
        "log_positive_prompt": "merged execution prompt",
        "log_negative_prompt": "",
        "positive": [],
        "negative": [],
        "styles": [],
        "task_seed": 7,
        "description": "route label, not user text",
    }

    save_and_log(
        task_state,
        64,
        64,
        [pytest.importorskip("numpy").zeros((8, 8, 3), dtype="uint8")],
        task_dict,
        False,
        [],
        workflow=workflow,
    )

    assert captured[expected_field] == expected_value
    assert captured[expected_field] != task_dict["description"]


def test_serializer_uses_only_frozen_active_controlnet_tasks(monkeypatch):
    captured = {}

    def fake_log(*args, **kwargs):
        captured.update(kwargs["clipboard_metadata"])
        return "captured.png"

    monkeypatch.setattr("modules.private_logger.log", fake_log)
    task_state = TaskState(
        prompt="main",
        save_metadata_to_images=False,
        workflow_plan=SimpleNamespace(route_id="txt2img"),
        cn_tasks={"PyraCanny": [["hidden-ui-slot"]]},
        planned_cn_tasks={"CPDS": [["active-frozen-slot"]]},
    )
    task_dict = {
        "log_positive_prompt": "main",
        "log_negative_prompt": "",
        "positive": [],
        "negative": [],
        "styles": [],
        "task_seed": 1,
    }

    save_and_log(
        task_state,
        64,
        64,
        [pytest.importorskip("numpy").zeros((8, 8, 3), dtype="uint8")],
        task_dict,
        False,
        [],
        workflow="txt2img",
    )

    assert captured["cn"] == [{"type": "CPDS"}]


def test_v1_conversion_does_not_execute_metadata(tmp_path):
    marker = tmp_path / "executed.txt"
    expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"

    converted = meta_parser.convert_v1_to_v2_metadata(
        {"Prompt": "safe", "Styles": expression}
    )

    assert not marker.exists()
    assert converted["styles"] == []


def test_v2_import_resolves_lora_stems_to_installed_filenames(monkeypatch):
    monkeypatch.setattr(config, "lora_filenames", ["portraits/face_detail.safetensors"])
    monkeypatch.setattr(
        meta_parser.modules.config,
        "lora_filenames",
        ["portraits/face_detail.safetensors"],
    )
    parser = meta_parser.FooocusMetadataParser()
    parsed = parser.to_json(
        {
            "metadata_version": 2,
            "workflow": "txt2img",
            "loras": [["face_detail", 0.65]],
        }
    )

    assert parsed["loras"] == [["portraits/face_detail.safetensors", 0.65]]


def test_partial_preset_does_not_synthesize_missing_metadata_fields():
    results = metadata_ui.load_preset_button_click(
        {"sampler": "euler"},
        is_generating=False,
    )

    assert results[metadata_ui.METADATA_OUTPUT_INDEX["sampler"]] == "euler"
    assert results[metadata_ui.METADATA_OUTPUT_INDEX["prompt"]] == {"__type__": "update"}
    assert results[metadata_ui.METADATA_OUTPUT_INDEX["steps"]] == {"__type__": "update"}
    assert results[metadata_ui.METADATA_OUTPUT_INDEX["base_model"]] == {"__type__": "update"}
    assert results[metadata_ui.METADATA_OUTPUT_INDEX["seed"]] == {"__type__": "update"}


def test_html_log_clipboard_uses_v2_when_image_embedding_is_disabled(tmp_path, monkeypatch):
    import args_manager
    import modules.private_logger as private_logger

    monkeypatch.setattr(config, "path_outputs", str(tmp_path))
    monkeypatch.setattr(private_logger.modules.config, "path_outputs", str(tmp_path))
    monkeypatch.setattr(args_manager.args, "disable_image_log", False, raising=False)
    private_logger.log_cache.clear()
    for index in range(101):
        private_logger.log_cache[f"old-{index}.html"] = "cached"

    task_state = TaskState(
        prompt="clipboard prompt",
        save_metadata_to_images=False,
        seed=12,
    )
    task_dict = {
        "log_positive_prompt": "clipboard prompt",
        "log_negative_prompt": "",
        "positive": ["clipboard prompt"],
        "negative": [],
        "styles": [],
        "task_seed": 12,
    }
    dummy_img = pytest.importorskip("numpy").zeros((8, 8, 3), dtype="uint8")

    save_and_log(task_state, 8, 8, [dummy_img], task_dict, False, [])

    log_paths = list(tmp_path.rglob("log.html"))
    assert len(log_paths) == 1
    html_text = log_paths[0].read_text(encoding="utf-8")
    payload_match = re.search(r"to_clipboard\(this, '([^']+)'\)", html_text)
    assert payload_match is not None
    payload = json.loads(urllib.parse.unquote(payload_match.group(1)))
    assert payload["metadata_version"] == 2
    assert payload["workflow"] == "txt2img"
    assert payload["version"] == "Nexfocus V1.0.0"
    assert "Nexfocus Image Log" in html_text
    assert len(private_logger.log_cache) == 100
