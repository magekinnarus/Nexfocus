import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)

sys.modules['args_manager'] = types.ModuleType('args_manager')
sys.modules['args_manager'].args = mock_args

from modules.model_download.policy import ModelDownloadPolicy
from modules.model_download.spec import ModelCatalogEntry, ModelSource


def _entry(root_key: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        id=f"entry.{root_key}",
        name='model.safetensors',
        root_key=root_key,
        relative_path='sdxl/base/model.safetensors',
        display_name='model',
        model_type='checkpoint',
        architecture='sdxl',
        sub_architecture='base',
        compatibility_family='sdxl',
        source_provider='huggingface',
        source=ModelSource(url='https://example.com/model.safetensors'),
        registration_state='sourced_registered',
        visibility='generic',
        preset_managed=False,
        token_required=False,
        tags=(),
    )


def test_model_download_policy_resolves_first_path_from_list(monkeypatch):
    from modules.model_download import policy as policy_mod
    def mock_get_preferred(*args, **kwargs):
        raise KeyError()
    monkeypatch.setattr(policy_mod.config, 'get_preferred_asset_root_path', mock_get_preferred)

    policy = ModelDownloadPolicy(root_map={'embeddings': ['D:/models/embeddings', 'E:/backup/embeddings']})

    resolved = policy.resolve_root_path(_entry('embeddings'))

    assert resolved == 'D:/models/embeddings'


def test_model_download_policy_rejects_empty_root_path_list(monkeypatch):
    from modules.model_download import policy as policy_mod
    def mock_get_preferred(*args, **kwargs):
        raise KeyError()
    monkeypatch.setattr(policy_mod.config, 'get_preferred_asset_root_path', mock_get_preferred)

    policy = ModelDownloadPolicy(root_map={'embeddings': []})

    try:
        policy.resolve_root_path(_entry('embeddings'))
    except KeyError as exc:
        assert 'No configured filesystem path' in str(exc)
    else:
        raise AssertionError('Expected KeyError for empty root path list')
