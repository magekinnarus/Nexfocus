import argparse
import json
import os
import sys
import types
import threading
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

from PIL import Image

from modules import model_thumbnails
from modules.model_download.spec import DownloadResult
from modules.model_manager import ModelManager


def _write_catalog(path: Path, payload: dict):
    normalized = json.loads(json.dumps(payload))
    for entry in normalized.get('entries', []):
        entry.setdefault('registration_state', 'sourced_registered')
        entry.setdefault('source', {'url': f"https://example.com/{entry['name']}"})
    path.write_text(json.dumps(normalized, indent=2), encoding='utf-8')




def test_model_manager_resolve_companion_clip_prefers_installed_clip(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    unet_dir = tmp_path / 'unet'
    clip_dir = tmp_path / 'clip'
    catalog_dir.mkdir()
    unet_dir.mkdir()
    clip_dir.mkdir()

    (unet_dir / 'sdxl' / 'illustrious').mkdir(parents=True, exist_ok=True)
    (clip_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (unet_dir / 'sdxl' / 'illustrious' / 'beretMixReal_v100.safetensors').write_text('unet', encoding='utf-8')
    (clip_dir / 'legacy' / 'IL_beret_textenc.safetensors').write_text('clip', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.unet.beret',
                    'name': 'beretMixReal_v100.safetensors',
                    'root_key': 'unet',
                    'relative_path': 'sdxl/illustrious/beretMixReal_v100.safetensors',
                    'model_type': 'unet',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'asset_group_key': 'illustrious.beretmix10',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.clip.beret',
                    'name': 'beretMixReal_v100_clips.safetensors',
                    'root_key': 'clip',
                    'relative_path': 'sdxl/illustrious/beretMixReal_v100_clips.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'asset_group_key': 'illustrious.beretmix10',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                },
                {
                    'id': 'unregistered.clip.beret',
                    'name': 'IL_beret_textenc.safetensors',
                    'root_key': 'clip',
                    'relative_path': 'legacy/IL_beret_textenc.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'unregistered',
                    'visibility': 'generic',
                    'tags': ['auto_generated', 'unregistered'],
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'unet': [unet_dir], 'clip': [clip_dir]},
    ).refresh()

    downloadable_companion = manager.resolve_companion_clip('sdxl/illustrious/beretMixReal_v100.safetensors')
    installed_companion = manager.resolve_companion_clip('sdxl/illustrious/beretMixReal_v100.safetensors', installed_only=True)

    assert downloadable_companion is not None
    assert downloadable_companion.id == 'entry.clip.beret'
    assert installed_companion is not None
    assert installed_companion.relative_path == 'legacy/IL_beret_textenc.safetensors'

def test_model_manager_discovers_installed_entries_and_groups_by_architecture(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    canonical_checkpoint = checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors'
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    canonical_checkpoint.write_text('checkpoint', encoding='utf-8')

    legacy_lora = loras_dir / 'pony_style.safetensors'
    legacy_lora.write_text('lora', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.sdxl.base.checkpoint',
                    'name': 'base_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/base_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.sdxl.pony.lora',
                    'name': 'pony_style.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/pony/pony_style.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'pony',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.sd15.checkpoint',
                    'name': 'sd15_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/sd15_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={
            'checkpoints': [checkpoints_dir],
            'loras': [loras_dir],
        },
    ).refresh()

    inventory = manager.build_inventory_payload()

    assert len(inventory['installed']) == 2
    assert len(inventory['available']) == 1
    assert inventory['groups'][0]['architecture'] == 'sd15'
    assert inventory['groups'][1]['architecture'] == 'sdxl'

    installed_ids = {record['id'] for record in inventory['installed']}
    assert 'entry.sdxl.base.checkpoint' in installed_ids
    assert 'entry.sdxl.pony.lora' in installed_ids


def test_model_manager_filters_dropdown_entries_by_architecture_and_preset_management(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (loras_dir / 'sdxl_base_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl_preset_lora.safetensors').write_text('lora', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.base.checkpoint',
                    'name': 'base_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/base_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.generic.lora',
                    'name': 'sdxl_base_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/sdxl_base_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.preset.lora',
                    'name': 'sdxl_preset_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/sdxl_preset_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'preset',
                    'preset_managed': True,
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={
            'checkpoints': [checkpoints_dir],
            'loras': [loras_dir],
        },
    ).refresh()

    entries = manager.list_dropdown_entries(base_model_name='base_model.safetensors', root_key='loras')

    assert [entry.entry.id for entry in entries] == ['entry.generic.lora']


def test_download_job_registry_tracks_progress_and_completion(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.downloadable',
                    'name': 'downloadable.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/downloadable.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
            ],
        },
    )

    def worker(entry, report_progress):
        assert entry.id == 'entry.downloadable'
        report_progress(0.25, 'starting')
        report_progress(1.0, 'finishing')
        return {
            'destination_path': str(checkpoints_dir / 'sdxl' / 'base' / 'downloadable.safetensors'),
            'message': 'done',
        }

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    job = manager.start_download_job('entry.downloadable', worker=worker)
    finished = manager.download_jobs.wait_for(job.job_id, timeout=5)

    assert finished is not None
    assert finished.status == 'succeeded'
    assert finished.progress == 1.0
    assert finished.result_path.endswith('downloadable.safetensors')
    assert finished.message == 'done'


def test_download_job_registry_marks_failed_results_as_failed(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.failed',
                    'name': 'failed.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/failed.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
            ],
        },
    )

    def worker(entry, report_progress):
        report_progress(0.25, 'starting')
        return DownloadResult(success=False, destination_path=str(checkpoints_dir / 'sdxl' / 'base' / 'failed.safetensors'), transport='test', message='download failed')

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    job = manager.start_download_job('entry.failed', worker=worker)
    finished = manager.download_jobs.wait_for(job.job_id, timeout=5)

    assert finished is not None
    assert finished.status == 'failed'
    assert finished.error == 'download failed'
    assert finished.message == 'download failed'




def test_download_job_registry_reuses_active_job_for_same_entry(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.downloadable',
                    'name': 'downloadable.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/downloadable.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
            ],
        },
    )

    started = threading.Event()
    release = threading.Event()
    calls = []

    def worker(entry, report_progress):
        calls.append(entry.id)
        report_progress(0.25, 'starting')
        started.set()
        assert release.wait(timeout=5)
        return {
            'destination_path': str(checkpoints_dir / 'sdxl' / 'base' / 'downloadable.safetensors'),
            'message': 'done',
        }

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    first_job = manager.start_download_job('entry.downloadable', worker=worker)
    assert started.wait(timeout=5)

    second_job = manager.start_download_job('entry.downloadable', worker=worker)
    assert second_job.job_id == first_job.job_id

    release.set()
    finished = manager.download_jobs.wait_for(first_job.job_id, timeout=5)

    assert calls == ['entry.downloadable']
    assert finished is not None
    assert finished.status == 'succeeded'
def test_model_manager_lists_filtered_installed_lora_choices_with_uncatalogued_entries(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'base' / 'catalogued_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base' / 'uncatalogued_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base' / 'sdxl_special_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sd15').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sd15' / 'sd15_lora.safetensors').write_text('lora', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.base.checkpoint',
                    'name': 'base_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/base_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.catalogued.lora',
                    'name': 'catalogued_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/catalogued_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={
            'checkpoints': [checkpoints_dir],
            'loras': [loras_dir],
        },
    ).refresh()

    choices = manager.list_installed_lora_dropdown_choices(base_model_name='base_model.safetensors')

    assert choices == [
        'sdxl/base/catalogued_lora.safetensors',
        'sdxl/base/sdxl_special_lora.safetensors',
        'sdxl/base/uncatalogued_lora.safetensors',
    ]


def test_model_manager_allows_sdxl_loras_across_sub_architectures(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'illustrious').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'pony').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'illustrious').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sd15' / 'base').mkdir(parents=True, exist_ok=True)

    (checkpoints_dir / 'sdxl' / 'illustrious' / 'illustrious_base.safetensors').write_text('checkpoint', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base' / 'base_style.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl' / 'pony' / 'pony_style.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl' / 'illustrious' / 'illustration_style.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sd15' / 'base' / 'sd15_style.safetensors').write_text('lora', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.base.checkpoint',
                    'name': 'illustrious_base.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/illustrious/illustrious_base.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.base.lora',
                    'name': 'base_style.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/base_style.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.pony.lora',
                    'name': 'pony_style.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/pony/pony_style.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'pony',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.illustrious.lora',
                    'name': 'illustration_style.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/illustrious/illustration_style.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.sd15.lora',
                    'name': 'sd15_style.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sd15/base/sd15_style.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir], 'loras': [loras_dir]},
    ).refresh()

    choices = manager.list_installed_lora_dropdown_choices(base_model_name='illustrious_base.safetensors')

    assert choices == [
        'sdxl/base/base_style.safetensors',
        'sdxl/illustrious/illustration_style.safetensors',
        'sdxl/pony/pony_style.safetensors',
    ]


def test_model_manager_persist_entry_thumbnail_updates_catalog_and_refreshes_index(tmp_path, monkeypatch):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    thumbnail_root = tmp_path / 'thumbnails'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    thumbnail_root.mkdir()

    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    default_path = thumbnail_root / 'default_0001.png'
    Image.new('RGB', (32, 32), color='blue').save(default_path)

    catalog_path = catalog_dir / 'runtime.catalog.json'
    _write_catalog(
        catalog_path,
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.base.checkpoint',
                    'alias': 'base_model',
                    'name': 'base_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/base_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'asset_group_key': 'base.base_model',
                    'thumbnail_library_relative': 'thumbnails/default_0001.png',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                    'preset_managed': False,
                    'token_required': False,
                    'source': {
                        'url': 'https://example.com/base_model.safetensors'
                    },
                }
            ],
        },
    )

    source_image = tmp_path / 'source.png'
    Image.new('RGB', (300, 200), color='red').save(source_image)

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    entry, resolution = manager.persist_entry_thumbnail('entry.base.checkpoint', source_image)
    persisted = json.loads(catalog_path.read_text(encoding='utf-8'))

    assert resolution.relative_path == 'thumbnails/checkpoints/sdxl/base/sdxl_base_checkpoint_base_model.png'
    assert entry.thumbnail_library_relative == resolution.relative_path
    assert persisted['entries'][0]['thumbnail_library_relative'] == resolution.relative_path
    assert manager.get_entry('entry.base.checkpoint').thumbnail_library_relative == resolution.relative_path



def test_model_manager_filters_by_registration_state(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.source.registered',
                    'name': 'remote_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/remote_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'registration_state': 'sourced_registered',
                },
                {
                    'id': 'entry.local.registered',
                    'name': 'local_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'legacy/local_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'locally_registered',
                    'source': {},
                },
            ],
        },
    )

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={'checkpoints': [checkpoints_dir]}).refresh()

    source_records = manager.iter_inventory(registration_state='sourced_registered')
    local_records = manager.iter_inventory(registration_state='locally_registered')

    assert [record.entry.id for record in source_records] == ['entry.source.registered']
    assert [record.entry.id for record in local_records] == ['entry.local.registered']



def test_model_manager_syncs_unregistered_installs_into_generated_catalog(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'mystery_model.safetensors').write_text('checkpoint', encoding='utf-8')

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    generated_path = catalog_dir / 'unregistered_install_catalog.catalog.json'
    payload = json.loads(generated_path.read_text(encoding='utf-8'))

    assert payload['catalog_id'] == 'user.unregistered.install'
    assert payload['entries'][0]['source_provider'] == 'local'
    assert payload['entries'][0]['registration_state'] == 'unregistered'
    assert payload['entries'][0]['relative_path'] == 'legacy/mystery_model.safetensors'

    records = manager.iter_inventory(registration_state='unregistered', installed=True)
    assert [record.entry.name for record in records] == ['mystery_model.safetensors']





def test_model_manager_prefers_authoritative_catalog_entry_over_stale_unregistered_draft(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'photon_v1.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.authoritative.photon',
                    'name': 'photon_v1.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/base/photon_v1.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'visibility': 'generic',
                },
            ],
        },
    )

    _write_catalog(
        catalog_dir / 'unregistered_install_catalog.catalog.json',
        {
            'catalog_id': 'user.unregistered.install',
            'entries': [
                {
                    'id': 'unregistered.checkpoints.photon',
                    'name': 'photon_v1.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'legacy/photon_v1.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'unregistered',
                    'visibility': 'generic',
                    'tags': ['auto_generated', 'unregistered'],
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    entry = manager.get_entry('legacy/PHOTON_v1.safetensors', root_keys=['checkpoints'])
    payload = json.loads((catalog_dir / 'unregistered_install_catalog.catalog.json').read_text(encoding='utf-8'))

    assert entry is not None
    assert entry.id == 'entry.authoritative.photon'
    assert payload['entries'] == []


def test_model_manager_refresh_prunes_stale_installed_links_for_missing_files(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.base.checkpoint',
                    'name': 'base_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/base_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
            ],
        },
    )

    missing_path = checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors'
    installed_links_path = catalog_dir / 'installed_model_links.json'
    installed_links_path.write_text(json.dumps({
        'catalog_id': 'user.installed.links',
        'catalog_label': 'Installed Model Links',
        'links': [
            {
                'install_id': 'deadbeef1234',
                'entry_id': 'entry.base.checkpoint',
                'root_key': 'checkpoints',
                'installed_path': str(missing_path),
                'installed_root_path': str(checkpoints_dir),
                'installed_relative_path': 'sdxl/base/base_model.safetensors',
            }
        ],
    }, indent=2), encoding='utf-8')

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    payload = json.loads(installed_links_path.read_text(encoding='utf-8'))

    assert manager._ensure_installed_links() == []
    assert payload['links'] == []



def test_model_manager_refresh_prunes_stale_installed_links_for_missing_entries(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    installed_path = checkpoints_dir / 'sdxl' / 'base' / 'ghost_model.safetensors'
    installed_path.parent.mkdir(parents=True, exist_ok=True)
    installed_path.write_text('ghost', encoding='utf-8')

    installed_links_path = catalog_dir / 'installed_model_links.json'
    installed_links_path.write_text(json.dumps({
        'catalog_id': 'user.installed.links',
        'catalog_label': 'Installed Model Links',
        'links': [
            {
                'install_id': 'ghost1234',
                'entry_id': 'entry.missing',
                'root_key': 'checkpoints',
                'installed_path': str(installed_path),
                'installed_root_path': str(checkpoints_dir),
                'installed_relative_path': 'sdxl/base/ghost_model.safetensors',
            }
        ],
    }, indent=2), encoding='utf-8')

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    payload = json.loads(installed_links_path.read_text(encoding='utf-8'))

    assert manager._ensure_installed_links() == []
    assert payload['links'] == []


def test_model_manager_build_installed_link_context_auto_creates_link_for_embedding(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    embeddings_dir = tmp_path / 'embeddings'
    catalog_dir.mkdir()
    embeddings_dir.mkdir()

    installed_path = embeddings_dir / 'easynegative.safetensors'
    installed_path.write_text('embedding', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'entries': [
                {
                    'id': 'entry.embedding.easynegative',
                    'name': 'easynegative.safetensors',
                    'root_key': 'embeddings',
                    'architecture': 'sd15',
                    'sub_architecture': 'none',
                    'model_type': 'embedding',
                    'compatibility_family': 'sd15',
                    'source_provider': 'civitai',
                    'source': {'url': 'https://example.com/easynegative.safetensors'},
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'embeddings': [embeddings_dir]},
    ).refresh()

    context = manager.build_installed_link_context('entry.embedding.easynegative')
    installed_links_payload = json.loads((catalog_dir / 'installed_model_links.json').read_text(encoding='utf-8'))

    assert context['entry']['sub_architecture'] == 'none'
    assert context['installed_link']['entry_id'] == 'entry.embedding.easynegative'
    assert context['installed_link']['installed_relative_path'] == 'easynegative.safetensors'
    assert installed_links_payload['links'][0]['entry_id'] == 'entry.embedding.easynegative'


def test_model_manager_installed_link_context_includes_resolved_thumbnail_for_blank_local_entry(tmp_path, monkeypatch):
    catalog_dir = tmp_path / 'catalogs'
    clip_dir = tmp_path / 'clip'
    thumbnail_root = tmp_path / 'thumbnails'
    catalog_dir.mkdir()
    clip_dir.mkdir()
    thumbnail_root.mkdir()

    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    Image.new('RGB', (32, 32), color='blue').save(thumbnail_root / 'default_0001.png')
    installed_path = clip_dir / 'clip_l.safetensors'
    installed_path.write_text('clip', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'user_local_models.catalog.json',
        {
            'catalog_id': 'user.local.models',
            'entries': [
                {
                    'id': 'user.local.clip.clip_l',
                    'name': 'clip_l.safetensors',
                    'display_name': 'clip l',
                    'root_key': 'clip',
                    'relative_path': 'clip_l.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'locally_registered',
                    'visibility': 'generic',
                    'thumbnail_library_relative': '',
                },
            ],
        },
    )
    generated_relative = model_thumbnails.build_thumbnail_relative_path(
        root_key='clip',
        architecture='sdxl',
        sub_architecture='base',
        model_type='clip',
        slug='clip_l.safetensors',
    )
    generated_path = model_thumbnails.resolve_thumbnail_absolute_path(generated_relative)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (64, 64), color='green').save(generated_path)

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'clip': [clip_dir]},
    ).refresh()

    context = manager.build_installed_link_context('user.local.clip.clip_l')

    assert context['entry']['thumbnail_library_relative'] == ''
    assert context['thumbnail']['relative_path'] == generated_relative
    assert context['thumbnail']['source'] == 'generated'

def test_model_manager_create_and_import_personal_catalogs(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()

    created = manager.create_personal_catalog(
        catalog_id='user.civitai.extra',
        catalog_label='User CivitAI Extra',
        source_provider='civitai',
        filename='civitai_extra',
    )

    assert created['catalog_id'] == 'user.civitai.extra'
    assert created['source_provider'] == 'civitai'
    assert created['file_name'] == 'civitai_extra.catalog.json'

    imported = manager.import_personal_catalog(
        {
            'catalog_id': 'user.huggingface.extra',
            'catalog_label': 'User HuggingFace Extra',
            'source_provider': 'huggingface',
            'entries': [
                {
                    'id': 'entry.imported.checkpoint',
                    'name': 'imported_model.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/imported_model.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                    'source': {'url': 'https://example.com/imported_model.safetensors'},
                }
            ],
        },
        filename='hf_extra.json',
    )

    assert imported['catalog_id'] == 'user.huggingface.extra'
    assert imported['file_name'] == 'hf_extra.catalog.json'
    assert manager.get_entry('entry.imported.checkpoint') is not None

    catalogs = manager.list_personal_catalogs()
    assert [catalog['catalog_id'] for catalog in catalogs] == [
        'user.civitai.extra',
        'user.huggingface.extra',
    ]

    try:
        manager.create_personal_catalog(
            catalog_id='user.civitai.extra',
            catalog_label='Duplicate CivitAI Extra',
            source_provider='civitai',
        )
    except ValueError as exc:
        assert 'Duplicate catalog id' in str(exc)
    else:
        raise AssertionError('Expected duplicate catalog id error')


def test_model_manager_register_model_entry_bundle_writes_to_target_personal_catalog(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'my_photon_copy.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'unregistered_install_catalog.catalog.json',
        {
            'catalog_id': 'user.unregistered.install',
            'catalog_label': 'Unregistered Installed Models',
            'entries': [
                {
                    'id': 'unregistered.checkpoints.photon',
                    'name': 'my_photon_copy.safetensors',
                    'display_name': 'my photon copy',
                    'root_key': 'checkpoints',
                    'relative_path': 'legacy/my_photon_copy.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'source_provider': 'local',
                    'registration_state': 'unregistered',
                    'visibility': 'generic',
                    'tags': ['auto_generated', 'unregistered'],
                    'source': {},
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()
    manager.create_personal_catalog(
        catalog_id='user.local.alt',
        catalog_label='Alternate Local Catalog',
        source_provider='local',
        filename='alt_local',
    )

    result = manager.register_model_entry_bundle(
        'legacy/my_photon_copy.safetensors',
        updates={
            'display_name': 'Photon Copy',
            'relative_path': 'sd15/base/my_photon_copy.safetensors',
            'architecture': 'sd15',
            'sub_architecture': 'base',
        },
        target_catalog_id='user.local.alt',
    )

    alt_catalog_payload = json.loads((catalog_dir / 'alt_local.catalog.json').read_text(encoding='utf-8'))

    assert result['entry'].id.startswith('user.local.checkpoints.')
    assert alt_catalog_payload['catalog_id'] == 'user.local.alt'
    assert [entry['id'] for entry in alt_catalog_payload['entries']] == [result['entry'].id]
    assert manager.get_entry(result['entry'].id) is not None



def test_model_manager_add_sourced_model_entry_creates_default_huggingface_catalog(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()

    result = manager.add_sourced_model_entry(
        source_provider='huggingface',
        source_input='https://huggingface.co/acme/repo/blob/main/models/my_model.safetensors?download=true',
        model_type='checkpoint',
        architecture='sdxl',
        sub_architecture='base',
    )

    payload = json.loads((catalog_dir / 'personal_download_catalog.catalog.json').read_text(encoding='utf-8'))

    assert result['catalog']['catalog_id'] == 'user.personal.default'
    assert result['entry'].id == 'huggingface.checkpoint.sdxl.base.my_model'
    assert result['entry'].source_provider == 'huggingface'
    assert result['entry'].source.url == 'https://huggingface.co/acme/repo/resolve/main/models/my_model.safetensors'
    assert payload['entries'][0]['asset_group_key'] == 'sdxl.base.my_model'
    assert payload['entries'][0]['relative_path'] == 'sdxl/base/my_model.safetensors'



def test_model_manager_add_sourced_model_entry_normalizes_civitai_identifier(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()
    manager.create_personal_catalog(
        catalog_id='user.civitai.extra',
        catalog_label='User CivitAI Extra',
        source_provider='civitai',
        filename='civitai_extra',
    )

    result = manager.add_sourced_model_entry(
        source_provider='civitai',
        source_input='https://civitai.com/models/12345?modelVersionId=67890',
        model_type='lora',
        architecture='sdxl',
        sub_architecture='illustrious',
        name='styleBoost.safetensors',
        target_catalog_id='user.civitai.extra',
    )

    payload = json.loads((catalog_dir / 'civitai_extra.catalog.json').read_text(encoding='utf-8'))

    assert result['catalog']['catalog_id'] == 'user.civitai.extra'
    assert result['entry'].source_provider == 'civitai'
    assert result['entry'].source_version_id == '67890'
    assert result['entry'].source.url == 'https://civitai.com/api/download/models/67890'
    assert result['entry'].token_required is True
    assert result['entry'].source.token_env == 'CIVITAI_TOKEN'
    assert payload['entries'][0]['id'] == result['entry'].id
