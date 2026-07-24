import argparse
import os
import sys
import types
from unittest.mock import MagicMock
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ui_logic = None

@pytest.fixture(scope="module", autouse=True)
def _setup_stubs_and_import_ui_logic():
    global ui_logic
    
    # Back up original modules to prevent polluting other tests
    modules_to_backup = [
        'args_manager',
        'backend.sampling',
        'backend.schedulers',
        'modules.async_worker',
        'shared',
        'modules.ui_components',
        'fooocus_version',
        'modules.html',
        'modules.gradio_hijack',
        'modules.style_sorter',
        'modules.meta_parser',
        'modules.objr_engine',
        'modules.ui_components.metadata_ui',
        'modules.ui_components.metadata_preview',
        'modules.ui_components.settings_panel',
        'modules.ui_components.styles_panel',
        'modules.ui_components.models_panel',
        'modules.ui_components.advanced_panel',
        'modules.ui_components.control_panel',
        'modules.ui_components.inpaint_panel',
        'modules.ui_components.outpaint_panel',
        'modules.private_logger',
        'modules.ui_gradio_extensions',
        'modules.auth',
        'modules.setup_utils',
    ]
    original_modules = {name: sys.modules.get(name) for name in modules_to_backup}

    # Set up stubs
    mock_args = argparse.Namespace()
    mock_args.colab = False
    mock_args.preset = None
    mock_args.output_path = None
    mock_args.temp_path = None
    mock_args.skip_model_load = False
    mock_args.disable_preset_selection = False
    mock_args.disable_image_log = False

    sys.modules['args_manager'] = MagicMock()
    sys.modules['args_manager'].args = mock_args

    sampling_stub = types.ModuleType('backend.sampling')
    sampling_stub.SAMPLER_NAMES = ['euler']
    schedulers_stub = types.ModuleType('backend.schedulers')
    schedulers_stub.SCHEDULER_NAMES = ['karras']
    sys.modules['backend.sampling'] = sampling_stub
    sys.modules['backend.schedulers'] = schedulers_stub

    async_worker_stub = types.ModuleType('modules.async_worker')
    async_worker_stub.AsyncTask = type('AsyncTask', (), {})
    async_worker_stub.async_tasks = []
    sys.modules['modules.async_worker'] = async_worker_stub

    shared_stub = types.ModuleType('shared')
    shared_stub.gradio_root = None
    sys.modules['shared'] = shared_stub

    ui_components_stub = types.ModuleType('modules.ui_components')
    sys.modules['modules.ui_components'] = ui_components_stub

    for module_name in [
        'fooocus_version',
        'modules.html',
        'modules.gradio_hijack',
        'modules.style_sorter',
        'modules.meta_parser',
        'modules.objr_engine',
        'modules.ui_components.metadata_ui',
        'modules.ui_components.metadata_preview',
        'modules.ui_components.settings_panel',
        'modules.ui_components.styles_panel',
        'modules.ui_components.models_panel',
        'modules.ui_components.advanced_panel',
        'modules.ui_components.control_panel',
        'modules.ui_components.inpaint_panel',
        'modules.ui_components.outpaint_panel',
        'modules.private_logger',
        'modules.ui_gradio_extensions',
        'modules.auth',
    ]:
        sys.modules[module_name] = types.ModuleType(module_name)

    sys.modules['modules.private_logger'].get_current_html_path = lambda output_format=None: 'history.html'
    sys.modules['modules.ui_gradio_extensions'].javascript_html = lambda: ''
    sys.modules['modules.ui_gradio_extensions'].css_html = lambda: ''
    sys.modules['modules.ui_components.metadata_preview'].format_metadata_preview = lambda *args, **kwargs: ''
    sys.modules['modules.auth'].auth_enabled = False
    sys.modules['modules.auth'].check_auth = lambda *args, **kwargs: True
    
    setup_utils_stub = types.ModuleType('modules.setup_utils')
    setup_utils_stub.download_models = lambda *args, **kwargs: (None, {}, False)
    setup_utils_stub.download_preset_models = lambda *args, **kwargs: (None, {}, False)
    sys.modules['modules.setup_utils'] = setup_utils_stub

    # Now import ui_logic so that it is initialized with our stubs in place
    import modules.ui_logic as imported_ui_logic
    ui_logic = imported_ui_logic

    yield

    # Restore original modules
    for name, orig in original_modules.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


class FakeModelManager:
    def list_installed_lora_dropdown_choices(self, *, base_model_name=None, include_preset_managed=False):
        assert base_model_name in {'base_model.safetensors', 'sdxl/base/base_model.safetensors'}
        choices = ['sdxl/base/compatible_lora.safetensors']
        if include_preset_managed:
            choices.append('sdxl/special/sdxl_special_lora.safetensors')
        return choices

    def resolve_companion_clip(self, selector_or_entry, installed_only=False):
        return None

    def get_entry(self, selector):
        entries = {
            'test.base': types.SimpleNamespace(root_key='checkpoints', relative_path='sdxl/base/base_model.safetensors', name='base_model.safetensors', architecture='sdxl'),
            'sdxl/base/base_model.safetensors': types.SimpleNamespace(root_key='checkpoints', relative_path='sdxl/base/base_model.safetensors', name='base_model.safetensors', architecture='sdxl'),
            'test.vae.sdxl': types.SimpleNamespace(root_key='vae', relative_path='sdxl/base/vae.safetensors', name='vae.safetensors', architecture='sdxl'),
            'test.vae.sd15': types.SimpleNamespace(root_key='vae', relative_path='sd15/base/vae_sd15.safetensors', name='vae_sd15.safetensors', architecture='sd15'),
            'test.clip.sdxl': types.SimpleNamespace(root_key='clip', relative_path='clip.safetensors', name='clip.safetensors', architecture='sdxl'),
            'test.clip.sd15': types.SimpleNamespace(root_key='clip', relative_path='clip_sd15.safetensors', name='clip_sd15.safetensors', architecture='sd15'),
        }
        return entries.get(selector)

    def inventory_record(self, entry):
        relative_path = getattr(entry, 'relative_path', None) or getattr(entry, 'name', None)
        return types.SimpleNamespace(installed=True, installed_relative_path=relative_path)


def test_update_model_dependent_choices_filters_incompatible_lora_values(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)', '1152x896 (9:7)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_compatible_vae_choices_for_model',
        lambda base_model_name: ['sdxl/base/vae.safetensors'],
    )
    updates = ui_logic.update_model_dependent_choices(
        'sdxl/base/base_model.safetensors',
        '1152x896 (9:7)',
        'sdxl/base/vae.safetensors',
        'sdxl/base/compatible_lora.safetensors',
        'sd15/incompatible_lora.safetensors',
    )

    assert updates[0]['choices'] == ['1024x1024 (1:1)', '1152x896 (9:7)']
    assert updates[0]['value'] == '1152x896 (9:7)'
    assert updates[1]['choices'] == [ui_logic.modules.flags.default_vae]
    assert updates[1]['value'] == ui_logic.modules.flags.default_vae
    assert updates[2]['choices'] == ['None', 'sdxl/base/compatible_lora.safetensors']
    assert updates[2]['value'] == 'sdxl/base/compatible_lora.safetensors'
    assert updates[3]['choices'] == ['None', 'sdxl/base/compatible_lora.safetensors']
    assert updates[3]['value'] == 'None'


def test_update_model_dependent_choices_preserves_active_preset_managed_lora(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)', '1152x896 (9:7)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_compatible_vae_choices_for_model',
        lambda base_model_name: ['sdxl/base/vae.safetensors'],
    )
    updates = ui_logic.update_model_dependent_choices(
        'sdxl/base/base_model.safetensors',
        '1152x896 (9:7)',
        'sdxl/base/vae.safetensors',
        'sdxl/special/sdxl_special_lora.safetensors',
    )

    assert updates[2]['choices'] == [
        'None',
        'sdxl/base/compatible_lora.safetensors',
        'sdxl/special/sdxl_special_lora.safetensors',
    ]
    assert updates[2]['value'] == 'sdxl/special/sdxl_special_lora.safetensors'


def test_update_model_dependent_choices_filters_incompatible_vae_choices(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_compatible_vae_choices_for_model',
        lambda base_model_name: ['sdxl/base/vae.safetensors'],
    )
    updates = ui_logic.update_model_dependent_choices(
        'sdxl/unet/test.safetensors',
        '1024x1024 (1:1)',
        'sd15/base/vae_sd15.safetensors',
        'None',
    )

    assert updates[1]['choices'] == [ui_logic.modules.flags.default_vae, 'sdxl/base/vae.safetensors']
    assert updates[1]['value'] == ui_logic.modules.flags.default_vae
    assert updates[2]['choices'] == ['None']
    assert updates[2]['value'] == 'None'


def test_update_model_dependent_choices_preserves_explicit_vae_for_unet_models(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )
    monkeypatch.setattr(ui_logic.modules.config, 'vae_filenames', ['sdxl/base/vae.safetensors'])

    class UnetModelManager(FakeModelManager):
        def get_entry(self, selector):
            entries = {
                'sdxl/unet/test.safetensors': types.SimpleNamespace(root_key='unet', relative_path='sdxl/unet/test.safetensors', name='test.safetensors', architecture='sdxl'),
            }
            return entries.get(selector) or super().get_entry(selector)

    monkeypatch.setattr(ui_logic, 'default_model_manager', UnetModelManager())

    updates = ui_logic.update_model_dependent_choices(
        'sdxl/unet/test.safetensors',
        '1024x1024 (1:1)',
        'sdxl/base/vae.safetensors',
        'None',
    )

    assert updates[1]['value'] == 'sdxl/base/vae.safetensors'


def test_apply_model_browser_drop_preserves_enabled_slot_when_lora_is_reset_after_base_switch(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(ui_logic.modules.config, 'model_filenames', ['sdxl/base/base_model.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'vae_filenames', ['vae.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'clip_filenames', ['clip.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'default_base_model_name', 'sdxl/base/base_model.safetensors')
    monkeypatch.setattr(ui_logic.modules.config, 'default_max_lora_number', 1)
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )

    updates = ui_logic.apply_model_browser_drop(
        '{"selector":"test.base","target":"base_model","aspect_ratio":"1024x1024 (1:1)","ts":1}',
        'sd15/base/old_model.safetensors',
        'None',
        True,
        'sd15/incompatible_lora.safetensors',
        1.0,
    )

    assert updates[0]['value'] == 'sdxl/base/base_model.safetensors'
    assert updates[3]['value'] is True
    assert updates[4]['value'] == 'None'


def test_apply_model_browser_drop_parses_json_bridge_payload(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(ui_logic.modules.config, 'model_filenames', ['sdxl/base/base_model.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'vae_filenames', ['vae.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'clip_filenames', ['clip.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'default_base_model_name', 'sdxl/base/base_model.safetensors')
    monkeypatch.setattr(ui_logic.modules.config, 'default_max_lora_number', 1)
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )

    updates = ui_logic.apply_model_browser_drop(
        '{"selector":"test.base","target":"base_model","aspect_ratio":"1024x1024 (1:1)","ts":1}',
        'old_model.safetensors',
        'None',
        False,
        'None',
        1.0,
    )

    assert updates[0]['value'] == 'sdxl/base/base_model.safetensors'


def test_apply_model_browser_drop_rejects_incompatible_vae(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(ui_logic.modules.config, 'model_filenames', ['sdxl/base/base_model.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'vae_filenames', ['sdxl/base/vae.safetensors', 'sd15/base/vae_sd15.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'clip_filenames', ['clip.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'default_base_model_name', 'sdxl/base/base_model.safetensors')
    monkeypatch.setattr(ui_logic.modules.config, 'default_max_lora_number', 1)
    monkeypatch.setattr(ui_logic.modules.config, 'get_aspect_ratio_labels_for_model', lambda base_model_name: ['1024x1024 (1:1)'])
    monkeypatch.setattr(ui_logic.modules.config, 'get_default_aspect_ratio_label_for_model', lambda base_model_name: '1024x1024 (1:1)')

    updates = ui_logic.apply_model_browser_drop(
        '{"selector":"test.vae.sd15","target":"vae_model","aspect_ratio":"1024x1024 (1:1)","ts":1}',
        'sdxl/base/base_model.safetensors',
        'sdxl/base/vae.safetensors',
        False,
        'None',
        1.0,
    )

    assert updates[2]['value'] == ui_logic.modules.flags.default_vae


def test_apply_model_browser_drop_refreshes_base_choices_when_checkpoint_is_installed_but_dropdown_is_stale(monkeypatch):
    monkeypatch.setattr(ui_logic, 'default_model_manager', FakeModelManager())
    monkeypatch.setattr(ui_logic.modules.config, 'model_filenames', ['sdxl/base/old_model.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'vae_filenames', ['vae.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'clip_filenames', ['clip.safetensors'])
    monkeypatch.setattr(ui_logic.modules.config, 'default_base_model_name', 'sdxl/base/base_model.safetensors')
    monkeypatch.setattr(ui_logic.modules.config, 'default_max_lora_number', 1)
    monkeypatch.setattr(ui_logic.modules.config, 'get_aspect_ratio_labels_for_model', lambda base_model_name: ['1024x1024 (1:1)'])
    monkeypatch.setattr(ui_logic.modules.config, 'get_default_aspect_ratio_label_for_model', lambda base_model_name: '1024x1024 (1:1)')

    refresh_calls = []

    def fake_refresh():
        refresh_calls.append('refresh')
        ui_logic.modules.config.model_filenames = ['sdxl/base/base_model.safetensors']

    monkeypatch.setattr(ui_logic, '_refresh_model_file_indexes', fake_refresh)

    updates = ui_logic.apply_model_browser_drop(
        '{"selector":"test.base","target":"base_model","aspect_ratio":"1024x1024 (1:1)","ts":1}',
        'sdxl/base/old_model.safetensors',
        'None',
        False,
        'None',
        1.0,
    )

    assert refresh_calls == ['refresh']
    assert updates[0]['choices'] == ['sdxl/base/base_model.safetensors']
    assert updates[0]['value'] == 'sdxl/base/base_model.safetensors'


def test_get_base_model_dropdown_state_falls_back_when_default_is_legacy_gguf(monkeypatch):
    monkeypatch.setattr(
        ui_logic.modules.config,
        'model_filenames',
        ['sdxl/base/base_model.safetensors'],
    )
    monkeypatch.setattr(
        ui_logic.modules.config,
        'default_base_model_name',
        'sdxl/illustrious/beretMixReal_v100_Q8.gguf',
    )

    choices, value = ui_logic._get_base_model_dropdown_state('sdxl/illustrious/beretMixReal_v100_Q8.gguf')

    assert choices == ['sdxl/base/base_model.safetensors']
    assert value == 'sdxl/base/base_model.safetensors'

def test_get_base_model_dropdown_state_returns_safe_none_choice_when_no_models(monkeypatch):
    monkeypatch.setattr(ui_logic.modules.config, 'model_filenames', [])
    monkeypatch.setattr(ui_logic.modules.config, 'default_base_model_name', 'None')

    choices, value = ui_logic._get_base_model_dropdown_state('missing_model.safetensors')

    assert choices == ['None']
    assert value == 'None'


def test_refresh_model_file_indexes_refreshes_catalog_before_rebuilding_checkpoint_choices(monkeypatch):
    events = []

    class RecordingManager:
        def refresh_catalog_index(self, *, force_refresh=False):
            events.append(('catalog', force_refresh))

        def refresh_installed_index(self):
            events.append(('installed', None))

    monkeypatch.setattr(ui_logic, 'default_model_manager', RecordingManager())
    monkeypatch.setattr(ui_logic.modules.config, 'update_files', lambda: events.append(('update_files', None)))

    ui_logic._refresh_model_file_indexes()

    assert events == [
        ('catalog', True),
        ('installed', None),
        ('update_files', None),
    ]


def test_coerce_active_base_model_selection_falls_back_when_default_is_sd15():
    choice = ui_logic.modules.config.coerce_active_base_model_selection(
        'sd15/base/legacy_model.safetensors',
        ['sdxl/base/base_model.safetensors'],
    )

    assert choice == 'sdxl/base/base_model.safetensors'

