import argparse
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = argparse.Namespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None
mock_args.skip_model_load = False

# Back up original modules to prevent polluting other tests
_MODULE_NAMES_TO_BACKUP = [
    'args_manager',
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _MODULE_NAMES_TO_BACKUP}

def teardown_module(module):
    # Restore original modules
    for name, orig in _ORIGINAL_MODULES.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig
        if '.' in name:
            parent_name, attr = name.rsplit('.', 1)
            parent_mod = sys.modules.get(parent_name)
            if parent_mod is not None:
                if orig is None:
                    try:
                        delattr(parent_mod, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(parent_mod, attr, orig)

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

from PIL import Image

from modules import model_thumbnails
from modules.model_download.spec import ModelCatalogEntry


def test_model_thumbnails_build_relative_path_for_sdxl_unet_family():
    entry = ModelCatalogEntry(
        id='entry.unet',
        alias='homoveritas_q4_k_m',
        name='homoveritasXL_v20NAIXLEPS.safetensors',
        root_key='unet',
        relative_path='sdxl/noob/homoveritasXL_v20NAIXLEPS.safetensors',
        model_type='unet',
        architecture='sdxl',
        sub_architecture='noob',
        asset_group_key='noob.homoveritas',
    )

    assert model_thumbnails.build_thumbnail_relative_path_for_entry(entry) == 'thumbnails/unet/sdxl/noob/sdxl_noob_unet_homoveritas.png'


def test_model_thumbnails_build_relative_path_for_sd15_and_vae_rules():
    checkpoint = ModelCatalogEntry(
        id='entry.checkpoint',
        alias='anything_v5',
        name='anything_v5.safetensors',
        root_key='checkpoints',
        relative_path='sd15/base/anything_v5.safetensors',
        model_type='checkpoint',
        architecture='sd15',
        sub_architecture='base',
    )
    vae = ModelCatalogEntry(
        id='entry.vae',
        alias='sdxl_vae',
        name='sdxl_vae.safetensors',
        root_key='vae',
        relative_path='sdxl/sdxl_vae.safetensors',
        model_type='vae',
        architecture='sdxl',
        sub_architecture='base',
    )

    assert model_thumbnails.build_thumbnail_relative_path_for_entry(checkpoint) == 'thumbnails/checkpoints/sd15/sd15_checkpoint_anything_v5.png'
    assert model_thumbnails.build_thumbnail_relative_path_for_entry(vae) == 'thumbnails/vae/sdxl/sdxl_vae_sdxl_vae.png'


def test_model_thumbnails_resolve_falls_back_to_default(tmp_path, monkeypatch):
    thumbnail_root = tmp_path / 'thumbnails'
    thumbnail_root.mkdir()
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    Image.new('RGB', (32, 32), color='blue').save(thumbnail_root / 'default_0001.png')
    entry = ModelCatalogEntry(
        id='entry.checkpoint',
        alias='anything_v5',
        name='anything_v5.safetensors',
        root_key='checkpoints',
        relative_path='sd15/base/anything_v5.safetensors',
        model_type='checkpoint',
        architecture='sd15',
        sub_architecture='base',
    )

    resolution = model_thumbnails.resolve_thumbnail(entry)

    assert resolution.relative_path == 'thumbnails/default_0001.png'
    assert resolution.exists is True
    assert resolution.source == 'default'


def test_model_thumbnails_persist_writes_square_png(tmp_path, monkeypatch):
    thumbnail_root = tmp_path / 'thumbnails'
    thumbnail_root.mkdir()
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 96)

    source = tmp_path / 'source.png'
    Image.new('RGB', (300, 200), color='red').save(source)
    entry = ModelCatalogEntry(
        id='entry.checkpoint',
        alias='anything_v5',
        name='anything_v5.safetensors',
        root_key='checkpoints',
        relative_path='sd15/base/anything_v5.safetensors',
        model_type='checkpoint',
        architecture='sd15',
        sub_architecture='base',
    )

    resolution = model_thumbnails.persist_thumbnail_image(source, entry=entry)
    persisted = Image.open(thumbnail_root / 'checkpoints' / 'sd15' / 'sd15_checkpoint_anything_v5.png')
    try:
        assert resolution.relative_path == 'thumbnails/checkpoints/sd15/sd15_checkpoint_anything_v5.png'
        assert persisted.size == (96, 60)
        assert persisted.format == 'PNG'
    finally:
        persisted.close()
