import json
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.model_download.catalog import load_model_catalog


def test_model_catalog_requires_source_object(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.missing.source',
                'name': 'model.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/model.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
                'source_provider': 'huggingface',
            }
        ],
    }), encoding='utf-8')

    with pytest.raises(ValueError, match="missing required 'source' metadata"):
        load_model_catalog(catalog_path)


def test_model_catalog_rejects_source_without_url(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.invalid.source',
                'name': 'model.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/model.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
                'source_provider': 'huggingface',
                'source': {},
            }
        ],
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='must define source.url'):
        load_model_catalog(catalog_path)



def test_model_catalog_defaults_registration_state_from_source_provider(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.default.state',
                'name': 'model.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/model.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
                'source_provider': 'huggingface',
                'source': {'url': 'https://example.com/model.safetensors'},
            }
        ],
    }), encoding='utf-8')

    entry = load_model_catalog(catalog_path).list()[0]

    assert entry.registration_state == 'sourced_registered'


def test_model_catalog_allows_local_entry_without_source(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.local.model',
                'name': 'model.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'legacy/model.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
                'source_provider': 'local',
                'registration_state': 'locally_registered',
            }
        ],
    }), encoding='utf-8')

    entry = load_model_catalog(catalog_path).list()[0]

    assert entry.registration_state == 'locally_registered'
    assert entry.source is None



def test_model_catalog_normalizes_singular_root_keys(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.singular.root',
                'name': 'model.safetensors',
                'root_key': 'checkpoint',
                'relative_path': 'sd15/base/model.safetensors',
                'architecture': 'sd15',
                'sub_architecture': 'base',
                'source_provider': 'huggingface',
                'source': {'url': 'https://example.com/model.safetensors'},
            }
        ],
    }), encoding='utf-8')

    entry = load_model_catalog(catalog_path).list()[0]

    assert entry.root_key == 'checkpoints'
    assert entry.model_type == 'checkpoint'


def test_model_catalog_derives_relative_path_for_sourced_entry(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.derived.path',
                'name': 'model.safetensors',
                'root_key': 'checkpoints',
                'architecture': 'sd15',
                'sub_architecture': 'base',
                'source_provider': 'huggingface',
                'source': {'url': 'https://example.com/model.safetensors'},
            }
        ],
    }), encoding='utf-8')

    entry = load_model_catalog(catalog_path).list()[0]

    assert entry.relative_path == 'sd15/base/model.safetensors'


def test_model_catalog_derives_relative_path_without_subfolder_for_vae_and_embeddings(tmp_path: Path):
    catalog_path = tmp_path / 'runtime.catalog.json'
    catalog_path.write_text(json.dumps({
        'catalog_id': 'runtime.catalog',
        'entries': [
            {
                'id': 'entry.vae.derived',
                'name': 'sdxl_vae.safetensors',
                'root_key': 'vae',
                'architecture': 'sdxl',
                'sub_architecture': 'none',
                'source_provider': 'huggingface',
                'source': {'url': 'https://example.com/sdxl_vae.safetensors'},
            },
            {
                'id': 'entry.embedding.derived',
                'name': 'unaestheticXL_neg.pt',
                'root_key': 'embeddings',
                'architecture': 'sdxl',
                'sub_architecture': 'none',
                'source_provider': 'civitai',
                'source': {'url': 'https://example.com/unaestheticXL_neg.pt'},
            }
        ],
    }), encoding='utf-8')

    entries = {entry.id: entry for entry in load_model_catalog(catalog_path).list()}

    assert entries['entry.vae.derived'].relative_path == 'sdxl/sdxl_vae.safetensors'
    assert entries['entry.vae.derived'].sub_architecture == 'none'
    assert entries['entry.embedding.derived'].relative_path == 'sdxl/unaestheticXL_neg.pt'
    assert entries['entry.embedding.derived'].sub_architecture == 'none'

