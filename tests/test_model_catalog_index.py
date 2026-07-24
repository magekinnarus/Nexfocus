import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.model_catalog_index import ModelCatalogIndex, is_runtime_catalog_file, iter_catalog_files


def _write_catalog(path: Path, catalog_id: str, entries: list[dict]):
    normalized_entries = []
    for entry in entries:
        normalized = dict(entry)
        normalized.setdefault('registration_state', 'sourced_registered')
        normalized.setdefault('source', {'url': f"https://example.com/{normalized['name']}"})
        normalized_entries.append(normalized)

    path.write_text(json.dumps({
        'catalog_id': catalog_id,
        'catalog_label': catalog_id,
        'entries': normalized_entries,
    }, indent=2), encoding='utf-8')


def test_iter_catalog_files_only_loads_runtime_catalogs(tmp_path):
    preset_dir = tmp_path / 'preset'
    preset_dir.mkdir()
    (preset_dir / 'alpha.catalog.json').write_text('{}', encoding='utf-8')
    (preset_dir / 'alpha_user_catalog.json').write_text('{}', encoding='utf-8')
    (preset_dir / 'beta.template.json').write_text('{}', encoding='utf-8')
    (preset_dir / 'gamma.example.json').write_text('{}', encoding='utf-8')

    paths = [path.name for path in iter_catalog_files([preset_dir])]

    assert paths == ['alpha.catalog.json', 'alpha_user_catalog.json']
    assert is_runtime_catalog_file(preset_dir / 'alpha.catalog.json') is True
    assert is_runtime_catalog_file(preset_dir / 'alpha_user_catalog.json') is True
    assert is_runtime_catalog_file(preset_dir / 'beta.template.json') is False


def test_model_catalog_index_merges_multiple_catalog_files(tmp_path):
    preset_dir = tmp_path / 'preset'
    user_dir = tmp_path / 'user'
    preset_dir.mkdir()
    user_dir.mkdir()

    _write_catalog(
        preset_dir / 'preset.catalog.json',
        'preset.catalog',
        [
            {
                'id': 'preset.checkpoint.base',
                'name': 'baseModel.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/baseModel.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
                'compatibility_family': 'sdxl',
            }
        ],
    )
    _write_catalog(
        user_dir / 'private.catalog.json',
        'private.catalog',
        [
            {
                'id': 'private.lora.pony',
                'alias': 'pony_lora',
                'name': 'ponyStyle.safetensors',
                'root_key': 'loras',
                'relative_path': 'sdxl/pony/ponyStyle.safetensors',
                'model_type': 'lora',
                'architecture': 'sdxl',
                'sub_architecture': 'pony',
            }
        ],
    )

    index = ModelCatalogIndex.from_directories([preset_dir, user_dir])

    assert [source.catalog_id for source in index.list_sources()] == ['preset.catalog', 'private.catalog']
    assert len(index.list()) == 2
    assert index.get('pony_lora').id == 'private.lora.pony'
    assert index.filter(compatibility_family='sdxl', model_type='lora')[0].sub_architecture == 'pony'


def test_model_catalog_index_raises_on_duplicate_ids(tmp_path):
    root = tmp_path / 'catalogs'
    root.mkdir()

    entry = {
        'id': 'duplicate.entry',
        'name': 'dup.safetensors',
        'root_key': 'checkpoints',
        'relative_path': 'sdxl/base/dup.safetensors',
        'model_type': 'checkpoint',
        'architecture': 'sdxl',
        'sub_architecture': 'base',
    }

    _write_catalog(root / 'a.catalog.json', 'a.catalog', [entry])
    _write_catalog(root / 'b.catalog.json', 'b.catalog', [dict(entry)])

    try:
        ModelCatalogIndex.from_directories([root])
    except ValueError as exc:
        assert 'Duplicate catalog id' in str(exc)
    else:
        raise AssertionError('Expected duplicate catalog id error')




def test_model_catalog_index_derives_relative_path_when_missing(tmp_path):
    root = tmp_path / 'catalogs'
    root.mkdir()

    _write_catalog(
        root / 'runtime.catalog.json',
        'runtime.catalog',
        [
            {
                'id': 'entry.derived.vae',
                'name': 'sdxl_vae.safetensors',
                'root_key': 'vae',
                'model_type': 'vae',
                'architecture': 'sdxl',
                'sub_architecture': 'none',
            }
        ],
    )

    index = ModelCatalogIndex.from_directories([root])

    assert index.find_by_relative_path('sdxl/sdxl_vae.safetensors', root_keys=['vae']).entry.id == 'entry.derived.vae'

def test_model_catalog_index_finds_entries_by_relative_path_and_name(tmp_path):
    root = tmp_path / 'catalogs'
    root.mkdir()

    _write_catalog(
        root / 'runtime.catalog.json',
        'runtime.catalog',
        [
            {
                'id': 'entry.pony.lora',
                'name': 'ponyStyle.safetensors',
                'root_key': 'loras',
                'relative_path': 'sdxl/pony/ponyStyle.safetensors',
                'model_type': 'lora',
                'architecture': 'sdxl',
                'sub_architecture': 'pony',
            }
        ],
    )

    index = ModelCatalogIndex.from_directories([root])

    assert index.find_by_relative_path(r'sdxl\pony\ponyStyle.safetensors', root_keys=['loras']).entry.id == 'entry.pony.lora'
    assert index.find_by_relative_path('SDXL/PONY/PONYSTYLE.SAFETENSORS', root_keys=['loras']).entry.id == 'entry.pony.lora'
    assert index.find_by_name('ponyStyle.safetensors', root_keys=['loras']).entry.id == 'entry.pony.lora'
    assert index.find_by_name('PONYSTYLE.SAFETENSORS', root_keys=['loras']).entry.id == 'entry.pony.lora'
import time

from modules.model_catalog_index import clear_runtime_model_catalog_index_cache, load_runtime_model_catalog_index


def test_model_catalog_index_requires_qualified_alias_for_collisions(tmp_path):
    preset_dir = tmp_path / 'preset'
    user_dir = tmp_path / 'user'
    preset_dir.mkdir()
    user_dir.mkdir()

    _write_catalog(
        preset_dir / 'preset.catalog.json',
        'preset.catalog',
        [
            {
                'id': 'preset.checkpoint.base',
                'alias': 'shared_alias',
                'name': 'baseModel.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/baseModel.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
            }
        ],
    )
    _write_catalog(
        user_dir / 'private.catalog.json',
        'private.catalog',
        [
            {
                'id': 'private.lora.pony',
                'alias': 'shared_alias',
                'name': 'ponyStyle.safetensors',
                'root_key': 'loras',
                'relative_path': 'sdxl/pony/ponyStyle.safetensors',
                'model_type': 'lora',
                'architecture': 'sdxl',
                'sub_architecture': 'pony',
            }
        ],
    )

    index = ModelCatalogIndex.from_directories([preset_dir, user_dir])

    try:
        index.get('shared_alias')
    except ValueError as exc:
        assert 'Ambiguous catalog alias' in str(exc)
    else:
        raise AssertionError('Expected ambiguous alias error')

    assert index.get('preset.catalog:shared_alias').id == 'preset.checkpoint.base'
    assert index.get('private.catalog:shared_alias').id == 'private.lora.pony'


def test_load_runtime_model_catalog_index_reuses_cache_until_catalogs_change(tmp_path):
    root = tmp_path / 'catalogs'
    root.mkdir()
    clear_runtime_model_catalog_index_cache()

    _write_catalog(
        root / 'runtime.catalog.json',
        'runtime.catalog',
        [
            {
                'id': 'entry.one',
                'name': 'one.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/one.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
            }
        ],
    )

    first = load_runtime_model_catalog_index([root])
    second = load_runtime_model_catalog_index([root])

    assert first is second

    time.sleep(0.02)
    _write_catalog(
        root / 'runtime.catalog.json',
        'runtime.catalog',
        [
            {
                'id': 'entry.two',
                'name': 'two.safetensors',
                'root_key': 'checkpoints',
                'relative_path': 'sdxl/base/two.safetensors',
                'model_type': 'checkpoint',
                'architecture': 'sdxl',
                'sub_architecture': 'base',
            }
        ],
    )

    third = load_runtime_model_catalog_index([root])

    assert third is not second
    assert third.get('entry.two').name == 'two.safetensors'
