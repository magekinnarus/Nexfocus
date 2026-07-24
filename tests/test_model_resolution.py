import argparse
import json
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

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

import modules.config as config


def _write_catalog(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def test_resolve_model_taxonomy_prefers_catalog_metadata_over_filename(tmp_path, monkeypatch):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()
    _write_catalog(
        catalog_dir / 'priority.catalog.json',
        {
            'catalog_id': 'priority.catalog',
            'entries': [
                {
                    'id': 'priority.sd15.model',
                    'name': 'XL_confusing_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/XL_confusing_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'source_provider': 'local',
                }
            ],
        },
    )

    monkeypatch.setattr(config, 'get_model_catalog_directories', lambda: [str(catalog_dir)])

    taxonomy = config.resolve_model_taxonomy('sd15/XL_confusing_model.safetensors', folder_paths=[str(tmp_path)])

    assert taxonomy.architecture == 'sd15'
    assert taxonomy.compatibility_family == 'sd15'
    assert taxonomy.source == 'catalog'
    assert config.get_default_aspect_ratio_for_model('sd15/XL_confusing_model.safetensors', folder_paths=[str(tmp_path)]) == '768*512'


def test_resolve_model_taxonomy_falls_back_to_filename_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'get_model_catalog_directories', lambda: [str(tmp_path / 'missing_catalogs')])

    taxonomy = config.resolve_model_taxonomy('IL_heroCharacter.safetensors', folder_paths=[str(tmp_path)])

    assert taxonomy.architecture == 'sdxl'
    assert taxonomy.sub_architecture == 'illustrious'
    assert taxonomy.compatibility_family == 'sdxl'
    assert taxonomy.source == 'filename'
    assert config.get_default_aspect_ratio_label_for_model('IL_heroCharacter.safetensors', folder_paths=[str(tmp_path)]) == config.add_ratio('1152*896')

def test_resolve_model_taxonomy_uses_path_segments_for_uncatalogued_sd15_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'get_model_catalog_directories', lambda: [str(tmp_path / 'missing_catalogs')])

    taxonomy = config.resolve_model_taxonomy('sd15/neutralModel.safetensors', folder_paths=[str(tmp_path)])

    assert taxonomy.architecture == 'sd15'
    assert taxonomy.sub_architecture == 'base'
    assert taxonomy.compatibility_family == 'sd15'
    assert taxonomy.source == 'filename'
    assert config.get_default_aspect_ratio_for_model('sd15/neutralModel.safetensors', folder_paths=[str(tmp_path)]) == '768*512'
