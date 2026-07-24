import argparse
from io import BytesIO
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
mock_args.listen = '127.0.0.1'
mock_args.port = 7860
mock_args.share = False
mock_args.in_browser = False

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from modules import model_thumbnails
from modules.model_api import create_model_router
from modules.model_download.spec import DownloadResult
from modules.model_manager import ModelManager


def _write_catalog(path: Path, payload: dict):
    normalized = json.loads(json.dumps(payload))
    for entry in normalized.get('entries', []):
        entry.setdefault('registration_state', 'sourced_registered')
        entry.setdefault('source', {'url': f"https://example.com/{entry['name']}"})
    path.write_text(json.dumps(normalized, indent=2), encoding='utf-8')


def _build_manager(tmp_path: Path) -> ModelManager:
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'base' / 'generic_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sd15').mkdir(parents=True, exist_ok=True)

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
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
                    'name': 'generic_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/generic_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.preset.lora',
                    'name': 'preset_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/preset_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'preset',
                    'preset_managed': True,
                },
                {
                    'id': 'entry.sd15.lora',
                    'name': 'sd15_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sd15/sd15_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'visibility': 'generic',
                },
            ],
        },
    )

    return ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={
            'checkpoints': [checkpoints_dir],
            'loras': [loras_dir],
        },
    ).refresh()


def test_model_api_catalog_browser_and_installed(tmp_path):
    manager = _build_manager(tmp_path)
    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    catalog = client.get('/api/models/catalog')
    assert catalog.status_code == 200
    assert catalog.json()['count'] == 4
    assert len(catalog.json()['sources']) == 1

    installed = client.get('/api/models/installed')
    assert installed.status_code == 200
    assert installed.json()['count'] == 2

    browser = client.get('/api/models/browser', params={'base_model_name': 'base_model.safetensors', 'root_key': 'loras'})
    assert browser.status_code == 200
    payload = browser.json()
    assert payload['scope']['architecture'] == 'sdxl'
    assert payload['scope']['sub_architecture'] == 'base'
    assert payload['installed'][0]['registration_state'] == 'sourced_registered'
    assert [entry['id'] for entry in payload['installed']] == ['entry.generic.lora']


def test_model_api_exposes_general_preset_loras_in_catalog_browser_and_installed(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (loras_dir / 'sdxl' / 'base' / 'sdxl_special_lora.safetensors').write_text('lora', encoding='utf-8')
    (loras_dir / 'sdxl' / 'base' / 'regular_lora.safetensors').write_text('lora', encoding='utf-8')

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
                    'id': 'entry.special.lora',
                    'name': 'sdxl_special_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/sdxl_special_lora.safetensors',
                    'model_type': 'lora',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'preset',
                    'preset_managed': True,
                },
                {
                    'id': 'entry.regular.lora',
                    'name': 'regular_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/regular_lora.safetensors',
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
    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    catalog = client.get('/api/models/catalog', params={'root_key': 'loras'})
    assert catalog.status_code == 200
    assert [entry['id'] for entry in catalog.json()['entries']] == ['entry.special.lora', 'entry.regular.lora']

    installed = client.get('/api/models/installed', params={'root_key': 'loras'})
    assert installed.status_code == 200
    assert [entry['id'] for entry in installed.json()['entries']] == ['entry.special.lora', 'entry.regular.lora']

    browser = client.get('/api/models/browser', params={'base_model_name': 'base_model.safetensors', 'root_key': 'loras'})
    assert browser.status_code == 200
    payload = browser.json()
    assert [entry['id'] for entry in payload['installed']] == ['entry.regular.lora']
    assert payload['available'] == []


def test_model_api_download_refreshes_installed_index(tmp_path):
    manager = _build_manager(tmp_path)
    app = FastAPI()

    def fake_download_worker(entry, report_progress):
        report_progress(0.5, 'working')
        root = Path(manager.root_map[entry.root_key][0])
        target = root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('downloaded', encoding='utf-8')
        return DownloadResult(success=True, destination_path=str(target), transport='test', message='done')

    app.include_router(create_model_router(manager, download_worker=fake_download_worker))
    client = TestClient(app)

    before = client.get('/api/models/installed')
    assert before.json()['count'] == 2

    response = client.post('/api/models/download', json={'selector': 'entry.sd15.lora'})
    assert response.status_code == 202
    job_id = response.json()['job']['job_id']

    job = manager.download_jobs.wait_for(job_id, timeout=5)
    assert job is not None
    assert job.status == 'succeeded'

    after = client.get('/api/models/installed')
    assert after.json()['count'] == 3
    assert any(entry['id'] == 'entry.sd15.lora' for entry in after.json()['entries'])

    refresh = client.post('/api/models/refresh')
    assert refresh.status_code == 200
    assert refresh.json()['installed'] == 3


def test_model_api_failed_download_reports_failed_status(tmp_path):
    manager = _build_manager(tmp_path)
    app = FastAPI()

    def fake_download_worker(entry, report_progress):
        report_progress(0.5, 'working')
        return DownloadResult(success=False, destination_path=str(Path(manager.root_map[entry.root_key][0]) / entry.relative_path), transport='test', message='auth failed')

    app.include_router(create_model_router(manager, download_worker=fake_download_worker))
    client = TestClient(app)

    response = client.post('/api/models/download', json={'selector': 'entry.sd15.lora'})
    assert response.status_code == 202
    job_id = response.json()['job']['job_id']

    job = manager.download_jobs.wait_for(job_id, timeout=5)
    assert job is not None
    assert job.status == 'failed'

    status = client.get(f'/api/models/downloads/{job_id}')
    assert status.status_code == 200
    assert status.json()['status'] == 'failed'
    assert status.json()['message'] == 'auth failed'


def test_model_api_thumbnail_resolve_and_persist(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    thumbnail_root = tmp_path / 'thumbs'
    thumbnail_root.mkdir()
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    default_path = thumbnail_root / 'default_0001.png'
    Image.new('RGB', (32, 32), color='blue').save(default_path)
    source_path = tmp_path / 'source.png'
    Image.new('RGB', (300, 200), color='red').save(source_path)

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    resolved = client.get('/api/models/thumbnail', params={'selector': 'entry.base.checkpoint'})
    assert resolved.status_code == 200
    assert resolved.json()['thumbnail']['relative_path'] == 'thumbnails/default_0001.png'
    assert resolved.json()['thumbnail']['source'] == 'default'

    persisted = client.post('/api/models/thumbnail', json={
        'selector': 'entry.base.checkpoint',
        'source_path': str(source_path),
    })
    assert persisted.status_code == 200
    payload = persisted.json()
    assert payload['status'] == 'success'
    assert payload['entry']['thumbnail_library_relative'] == 'thumbnails/checkpoints/sdxl/base/sdxl_base_checkpoint_base_model.png'
    assert payload['thumbnail']['exists'] is True

    resolved_after = client.get('/api/models/thumbnail', params={'selector': 'entry.base.checkpoint'})
    assert resolved_after.status_code == 200
    assert resolved_after.json()['thumbnail']['source'] == 'catalog'
    assert resolved_after.json()['thumbnail']['relative_path'] == 'thumbnails/checkpoints/sdxl/base/sdxl_base_checkpoint_base_model.png'







def test_model_api_thumbnail_file_serves_image(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    thumbnail_root = tmp_path / 'thumbs'
    thumbnail_root.mkdir()
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    default_path = thumbnail_root / 'default_0001.png'
    Image.new('RGB', (32, 32), color='blue').save(default_path)

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    response = client.get('/api/models/thumbnail/file', params={'selector': 'entry.base.checkpoint'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('image/')
    assert response.content

def test_model_api_registration_context_and_save(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'my_photon_copy.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.authoritative.photon',
                    'name': 'photon_v1.safetensors',
                    'display_name': 'Photon V1',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/base/photon_v1.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'source_provider': 'civitai',
                    'source_version_id': '12345',
                    'visibility': 'generic',
                },
            ],
        },
    )
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
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
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

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    context = client.get('/api/models/registration', params={
        'selector': 'legacy/my_photon_copy.safetensors',
        'source_provider': 'civitai',
        'source_version_id': '12345',
    })
    assert context.status_code == 200
    payload = context.json()
    assert payload['entry']['registration_state'] == 'unregistered'
    assert payload['suggestions'][0]['entry']['id'] == 'entry.authoritative.photon'
    assert payload['suggestions'][0]['entry']['source_version_id'] == '12345'

    registered = client.post('/api/models/registration', json={
        'selector': 'legacy/my_photon_copy.safetensors',
        'matched_selector': 'entry.authoritative.photon',
        'updates': {
            'display_name': 'Photon V1',
        },
    })
    assert registered.status_code == 200
    registered_payload = registered.json()
    assert registered_payload['entry']['registration_state'] == 'sourced_registered'
    assert registered_payload['entry']['display_name'] == 'Photon V1'
    assert registered_payload['entry']['name'] == 'photon_v1.safetensors'
    assert registered_payload['entry']['source_provider'] == 'civitai'
    assert registered_payload['entry']['source_version_id'] == '12345'





def test_model_api_installed_link_context_and_relink(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'my_photon_copy.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.authoritative.photon',
                    'name': 'photon_v1.safetensors',
                    'display_name': 'Photon V1',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/base/photon_v1.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'source_provider': 'civitai',
                    'source_version_id': '12345',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.authoritative.photon.alt',
                    'name': 'photon_v2.safetensors',
                    'display_name': 'Photon V2',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/base/photon_v2.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sd15',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sd15',
                    'source_provider': 'civitai',
                    'source_version_id': '67890',
                    'visibility': 'generic',
                },
            ],
        },
    )
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

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    first_registration = client.post('/api/models/registration', json={
        'selector': 'legacy/my_photon_copy.safetensors',
        'matched_selector': 'entry.authoritative.photon',
    })
    assert first_registration.status_code == 200

    context = client.get('/api/models/installed-link', params={'selector': 'entry.authoritative.photon'})
    assert context.status_code == 200
    context_payload = context.json()
    assert context_payload['mode'] == 'installed_link'
    assert context_payload['installed_link']['entry_id'] == 'entry.authoritative.photon'
    assert context_payload['installed_link']['installed_relative_path'] == 'legacy/my_photon_copy.safetensors'

    relinked = client.post('/api/models/installed-link', json={
        'selector': 'entry.authoritative.photon',
        'matched_selector': 'entry.authoritative.photon.alt',
        'updates': {
            'installed_relative_path': 'legacy/my_photon_copy.safetensors',
        },
    })
    assert relinked.status_code == 200
    relinked_payload = relinked.json()
    assert relinked_payload['entry']['id'] == 'entry.authoritative.photon.alt'
    assert relinked_payload['installed_link']['entry_id'] == 'entry.authoritative.photon.alt'
    assert relinked_payload['installed_link']['installed_relative_path'] == 'legacy/my_photon_copy.safetensors'

    installed = client.get('/api/models/installed', params={'root_key': 'checkpoints'})
    assert installed.status_code == 200
    installed_entries = {entry['id']: entry for entry in installed.json()['entries']}
    assert installed_entries['entry.authoritative.photon.alt']['installed_relative_path'] == 'legacy/my_photon_copy.safetensors'
    assert 'entry.authoritative.photon' not in installed_entries
def test_model_api_registration_context_includes_companion_clip_for_unet(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    unet_dir = tmp_path / 'unet'
    clip_dir = tmp_path / 'clip'
    catalog_dir.mkdir()
    unet_dir.mkdir()
    clip_dir.mkdir()

    (unet_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (clip_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (unet_dir / 'legacy' / 'IL_beret.safetensors').write_text('unet', encoding='utf-8')
    (clip_dir / 'legacy' / 'IL_beret_textenc.safetensors').write_text('clip', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.authoritative.unet.beret',
                    'name': 'beretMixReal_v100.safetensors',
                    'display_name': 'BeretMixReal v100 Q8',
                    'root_key': 'unet',
                    'relative_path': 'sdxl/illustrious/beretMixReal_v100.safetensors',
                    'model_type': 'unet',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'asset_group_key': 'illustrious.beretmix10',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                    'source': {'url': 'https://example.invalid/beretMixReal_v100.safetensors'},
                },
                {
                    'id': 'entry.authoritative.clip.beret',
                    'name': 'beretMixReal_v100_clips.safetensors',
                    'display_name': 'BeretMixReal v100 clips',
                    'root_key': 'clip',
                    'relative_path': 'sdxl/illustrious/beretMixReal_v100_clips.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'asset_group_key': 'illustrious.beretmix10',
                    'source_provider': 'huggingface',
                    'visibility': 'generic',
                    'source': {'url': 'https://example.invalid/beretMixReal_v100_clips.safetensors'},
                },
            ],
        },
    )
    _write_catalog(
        catalog_dir / 'unregistered_install_catalog.catalog.json',
        {
            'catalog_id': 'user.unregistered.install',
            'catalog_label': 'Unregistered Installed Models',
            'entries': [
                {
                    'id': 'unregistered.unet.beret',
                    'name': 'IL_beret.safetensors',
                    'display_name': 'IL beret q8',
                    'root_key': 'unet',
                    'relative_path': 'legacy/IL_beret.safetensors',
                    'model_type': 'unet',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'unregistered',
                    'visibility': 'generic',
                    'tags': ['auto_generated', 'unregistered'],
                },
                {
                    'id': 'unregistered.clip.beret',
                    'name': 'IL_beret_textenc.safetensors',
                    'display_name': 'IL beret textenc',
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

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    context = client.get('/api/models/registration', params={
        'selector': 'legacy/IL_beret.safetensors',
        'matched_selector': 'entry.authoritative.unet.beret',
    })
    assert context.status_code == 200
    companion_context = context.json()['companion_clip']
    assert companion_context['matched_catalog_entry']['id'] == 'entry.authoritative.clip.beret'
    assert companion_context['installed_candidates'][0]['entry']['name'] == 'IL_beret_textenc.safetensors'

    registered = client.post('/api/models/registration', json={
        'selector': 'legacy/IL_beret.safetensors',
        'matched_selector': 'entry.authoritative.unet.beret',
    })
    assert registered.status_code == 200
    payload = registered.json()
    assert payload['entry']['name'] == 'beretMixReal_v100.safetensors'
    assert payload['installed_link']['entry_id'] == 'entry.authoritative.unet.beret'
    assert payload['installed_link']['installed_relative_path'] == 'legacy/IL_beret.safetensors'
    assert 'companion_clip' not in payload

    unregistered_payload = json.loads((catalog_dir / 'unregistered_install_catalog.catalog.json').read_text(encoding='utf-8'))
    assert len(unregistered_payload['entries']) == 1
    assert unregistered_payload['entries'][0]['root_key'] == 'clip'
    assert unregistered_payload['entries'][0]['relative_path'] == 'legacy/IL_beret_textenc.safetensors'



def test_model_api_registration_creates_local_registered_catalog_and_installed_link(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'mystery_model.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'unregistered_install_catalog.catalog.json',
        {
            'catalog_id': 'user.unregistered.install',
            'catalog_label': 'Unregistered Installed Models',
            'entries': [
                {
                    'id': 'unregistered.checkpoints.mystery',
                    'name': 'mystery_model.safetensors',
                    'display_name': 'mystery model',
                    'root_key': 'checkpoints',
                    'relative_path': 'legacy/mystery_model.safetensors',
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

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    registered = client.post('/api/models/registration', json={
        'selector': 'legacy/mystery_model.safetensors',
        'updates': {
            'name': 'studioMystery_v1.safetensors',
            'display_name': 'Studio Mystery v1',
            'model_type': 'checkpoint',
            'architecture': 'sd15',
            'sub_architecture': 'base',
            'compatibility_family': 'sd15',
            'source_provider': 'local',
        },
    })
    assert registered.status_code == 200
    payload = registered.json()
    assert payload['entry']['id'].startswith('user.local.checkpoints.')
    assert payload['entry']['registration_state'] == 'locally_registered'
    assert payload['entry']['name'] == 'studioMystery_v1.safetensors'
    assert payload['entry']['installed_relative_path'] == 'legacy/mystery_model.safetensors'
    assert payload['installed_link']['entry_id'] == payload['entry']['id']

    local_catalog_payload = json.loads((catalog_dir / 'user_local_models.catalog.json').read_text(encoding='utf-8'))
    assert local_catalog_payload['entries'][0]['id'] == payload['entry']['id']
    assert local_catalog_payload['entries'][0]['name'] == 'studioMystery_v1.safetensors'

    installed_links_payload = json.loads((catalog_dir / 'installed_model_links.json').read_text(encoding='utf-8'))
    assert installed_links_payload['links'][0]['entry_id'] == payload['entry']['id']
    assert installed_links_payload['links'][0]['installed_relative_path'] == 'legacy/mystery_model.safetensors'

    unregistered_payload = json.loads((catalog_dir / 'unregistered_install_catalog.catalog.json').read_text(encoding='utf-8'))
    assert unregistered_payload['entries'] == []

    lookup = client.get('/api/models/registration', params={'selector': 'legacy/mystery_model.safetensors'})
    assert lookup.status_code == 200
    assert lookup.json()['entry']['id'] == payload['entry']['id']
def test_model_api_batch_download_limits_to_active_root_key(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    loras_dir = tmp_path / 'loras'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()
    loras_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.available.checkpoint',
                    'name': 'remote_checkpoint.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sdxl/base/remote_checkpoint.safetensors',
                    'model_type': 'checkpoint',
                    'architecture': 'sdxl',
                    'sub_architecture': 'base',
                    'compatibility_family': 'sdxl',
                    'visibility': 'generic',
                },
                {
                    'id': 'entry.available.lora',
                    'name': 'remote_lora.safetensors',
                    'root_key': 'loras',
                    'relative_path': 'sdxl/base/remote_lora.safetensors',
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

    def fake_download_worker(entry, report_progress):
        report_progress(0.5, 'working')
        root = Path(manager.root_map[entry.root_key][0])
        target = root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('downloaded', encoding='utf-8')
        return DownloadResult(success=True, destination_path=str(target), transport='test', message='done')

    app = FastAPI()
    app.include_router(create_model_router(manager, download_worker=fake_download_worker))
    client = TestClient(app)

    response = client.post('/api/models/downloads/batch', json={
        'root_key': 'loras',
        'selectors': ['entry.available.checkpoint', 'entry.available.lora'],
    })
    assert response.status_code == 202
    payload = response.json()
    assert payload['queued_count'] == 1
    assert payload['jobs'][0]['entry_id'] == 'entry.available.lora'
    assert payload['skipped'] == [{'selector': 'entry.available.checkpoint', 'reason': 'different_root_key'}]

    job_id = payload['jobs'][0]['job_id']
    job = manager.download_jobs.wait_for(job_id, timeout=5)
    assert job is not None
    assert job.status == 'succeeded'



def test_model_api_batch_download_for_unet_also_queues_companion_clip(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    unet_dir = tmp_path / 'unet'
    clip_dir = tmp_path / 'clip'
    catalog_dir.mkdir()
    unet_dir.mkdir()
    clip_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.available.unet',
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
                    'id': 'entry.available.clip',
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
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'unet': [unet_dir], 'clip': [clip_dir]},
    ).refresh()

    def fake_download_worker(entry, report_progress):
        report_progress(0.5, 'working')
        root = Path(manager.root_map[entry.root_key][0])
        target = root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('downloaded', encoding='utf-8')
        return DownloadResult(success=True, destination_path=str(target), transport='test', message='done')

    app = FastAPI()
    app.include_router(create_model_router(manager, download_worker=fake_download_worker))
    client = TestClient(app)

    response = client.post('/api/models/downloads/batch', json={
        'root_key': 'unet',
        'selectors': ['entry.available.unet'],
    })
    assert response.status_code == 202
    payload = response.json()
    queued_ids = [job['entry_id'] for job in payload['jobs']]
    assert queued_ids == ['entry.available.unet', 'entry.available.clip']

def test_model_api_browser_splits_registered_and_unregistered_installed(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (checkpoints_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'legacy' / 'mystery_model.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
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

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    browser = client.get('/api/models/browser', params={'root_key': 'checkpoints'})
    assert browser.status_code == 200
    payload = browser.json()

    assert [entry['name'] for entry in payload['installed_registered']] == ['base_model.safetensors']
    assert [entry['name'] for entry in payload['installed_unregistered']] == ['mystery_model.safetensors']

    refresh = client.post('/api/models/refresh')
    assert refresh.status_code == 200
    assert refresh.json()['unregistered_installed'] == 1


def test_model_api_browser_hides_sd15_checkpoint_records_by_default_but_preserves_explicit_query(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    checkpoints_dir = tmp_path / 'checkpoints'
    catalog_dir.mkdir()
    checkpoints_dir.mkdir()

    (checkpoints_dir / 'sdxl' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sdxl' / 'base' / 'base_model.safetensors').write_text('checkpoint', encoding='utf-8')
    (checkpoints_dir / 'sd15' / 'base').mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / 'sd15' / 'base' / 'legacy_sd15.safetensors').write_text('checkpoint', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
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
                    'id': 'entry.legacy.sd15',
                    'name': 'legacy_sd15.safetensors',
                    'root_key': 'checkpoints',
                    'relative_path': 'sd15/base/legacy_sd15.safetensors',
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
        root_map={'checkpoints': [checkpoints_dir]},
    ).refresh()

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    browser = client.get('/api/models/browser', params={'root_key': 'checkpoints'})
    assert browser.status_code == 200
    payload = browser.json()

    assert [entry['name'] for entry in payload['installed_registered']] == ['base_model.safetensors']
    assert [group['architecture'] for group in payload['groups']] == ['sdxl']

    explicit_sd15 = client.get('/api/models/browser', params={'root_key': 'checkpoints', 'architecture': 'sd15'})
    assert explicit_sd15.status_code == 200
    explicit_payload = explicit_sd15.json()

    assert [entry['name'] for entry in explicit_payload['installed_registered']] == ['legacy_sd15.safetensors']







def test_model_api_registration_forces_none_sub_architecture_for_embeddings_and_persists_thumbnail(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    embeddings_dir = tmp_path / 'embeddings'
    catalog_dir.mkdir()
    embeddings_dir.mkdir()

    (embeddings_dir / 'legacy').mkdir(parents=True, exist_ok=True)
    (embeddings_dir / 'legacy' / 'easynegative_copy.safetensors').write_text('embedding', encoding='utf-8')

    _write_catalog(
        catalog_dir / 'unregistered_install_catalog.catalog.json',
        {
            'catalog_id': 'user.unregistered.install',
            'catalog_label': 'Unregistered Installed Models',
            'entries': [
                {
                    'id': 'unregistered.embeddings.easynegative',
                    'name': 'easynegative_copy.safetensors',
                    'display_name': 'easynegative copy',
                    'root_key': 'embeddings',
                    'relative_path': 'legacy/easynegative_copy.safetensors',
                    'model_type': 'embedding',
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
        root_map={'embeddings': [embeddings_dir]},
    ).refresh()

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    registered = client.post('/api/models/registration', json={
        'selector': 'legacy/easynegative_copy.safetensors',
        'updates': {
            'display_name': 'EasyNegative Copy',
            'name': 'easynegative_copy.safetensors',
            'model_type': 'embedding',
            'architecture': 'sd15',
            'sub_architecture': 'general',
            'thumbnail_library_relative': 'thumbnails/embeddings/sd15/sd15_embedding_easynegative_copy.png',
        },
    })
    assert registered.status_code == 200
    payload = registered.json()
    assert payload['entry']['sub_architecture'] == 'none'
    assert payload['entry']['thumbnail_library_relative'] == 'thumbnails/embeddings/sd15/sd15_embedding_easynegative_copy.png'
    assert payload['entry']['registration_state'] == 'locally_registered'

    local_catalog = json.loads((catalog_dir / 'user_local_models.catalog.json').read_text(encoding='utf-8'))
    assert local_catalog['entries'][0]['sub_architecture'] == 'none'
    assert local_catalog['entries'][0]['thumbnail_library_relative'] == 'thumbnails/embeddings/sd15/sd15_embedding_easynegative_copy.png'


def test_model_api_browser_lists_embedding_entries(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.embedding.easynegative',
                    'name': 'easynegative.safetensors',
                    'root_key': 'embeddings',
                    'model_type': 'embedding',
                    'architecture': 'sd15',
                    'sub_architecture': 'none',
                    'compatibility_family': 'sd15',
                    'source_provider': 'civitai',
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'embeddings': []},
    ).refresh()

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    browser = client.get('/api/models/browser', params={'root_key': 'embeddings', 'architecture': 'sd15'})
    assert browser.status_code == 200
    payload = browser.json()
    assert [entry['id'] for entry in payload['available_registered']] == ['entry.embedding.easynegative']


def test_model_api_thumbnail_upload_persists_image(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    thumbnail_root = tmp_path / 'thumbs'
    thumbnail_root.mkdir()
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_directory', lambda: str(thumbnail_root))
    monkeypatch.setattr(model_thumbnails.config, 'get_default_thumbnail_relative_path', lambda: 'thumbnails/default_0001.png')
    monkeypatch.setattr(model_thumbnails.config, 'get_model_thumbnail_size', lambda: 128)

    default_path = thumbnail_root / 'default_0001.png'
    Image.new('RGB', (32, 32), color='blue').save(default_path)

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    upload_bytes = BytesIO()
    Image.new('RGB', (256, 192), color='green').save(upload_bytes, format='PNG')
    upload_bytes.seek(0)

    response = client.post(
        '/api/models/thumbnail/upload',
        files={'file': ('upload.png', upload_bytes, 'image/png')},
        data={'selector': 'entry.base.checkpoint'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'success'
    assert payload['entry']['thumbnail_library_relative'] == 'thumbnails/checkpoints/sdxl/base/sdxl_base_checkpoint_base_model.png'
    assert payload['thumbnail']['exists'] is True


def test_model_api_browser_excludes_local_only_registered_models_from_available(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    _write_catalog(
        catalog_dir / 'runtime.catalog.json',
        {
            'catalog_id': 'runtime.catalog',
            'catalog_label': 'Runtime',
            'entries': [
                {
                    'id': 'entry.local.only',
                    'name': 'local_only.safetensors',
                    'root_key': 'clip',
                    'relative_path': 'legacy/local_only.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'local',
                    'registration_state': 'locally_registered',
                    'visibility': 'generic',
                    'source': {},
                },
                {
                    'id': 'entry.remote.clip',
                    'name': 'remote_clip.safetensors',
                    'root_key': 'clip',
                    'relative_path': 'sdxl/illustrious/remote_clip.safetensors',
                    'model_type': 'clip',
                    'architecture': 'sdxl',
                    'sub_architecture': 'illustrious',
                    'compatibility_family': 'sdxl',
                    'source_provider': 'huggingface',
                    'registration_state': 'sourced_registered',
                    'visibility': 'generic',
                },
            ],
        },
    )

    manager = ModelManager(
        catalog_dirs=[catalog_dir],
        root_map={'clip': []},
    ).refresh()

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    browser = client.get('/api/models/browser', params={'root_key': 'clip'})
    assert browser.status_code == 200
    payload = browser.json()
    assert [entry['id'] for entry in payload['available_registered']] == ['entry.remote.clip']

def test_model_api_personal_catalog_endpoints(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()
    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    created = client.post('/api/models/personal-catalogs', json={
        'catalog_id': 'user.civitai.extra',
        'catalog_label': 'User CivitAI Extra',
        'source_provider': 'civitai',
        'filename': 'civitai_extra',
    })
    assert created.status_code == 200
    assert created.json()['catalog']['file_name'] == 'civitai_extra.catalog.json'

    imported = client.post('/api/models/personal-catalogs/import', json={
        'filename': 'hf_extra.json',
        'catalog': {
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
    })
    assert imported.status_code == 200
    assert imported.json()['catalog']['file_name'] == 'hf_extra.catalog.json'

    listed = client.get('/api/models/personal-catalogs')
    assert listed.status_code == 200
    assert listed.json()['count'] == 2
    assert [catalog['catalog_id'] for catalog in listed.json()['catalogs']] == [
        'user.civitai.extra',
        'user.huggingface.extra',
    ]


def test_model_api_registration_supports_target_personal_catalog(tmp_path):
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

    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    response = client.post('/api/models/registration', json={
        'selector': 'legacy/my_photon_copy.safetensors',
        'target_catalog_id': 'user.local.alt',
        'updates': {
            'display_name': 'Photon Copy',
            'relative_path': 'sd15/base/my_photon_copy.safetensors',
            'architecture': 'sd15',
            'sub_architecture': 'base',
        },
    })
    assert response.status_code == 200

    alt_catalog_payload = json.loads((catalog_dir / 'alt_local.catalog.json').read_text(encoding='utf-8'))
    assert response.json()['entry']['id'].startswith('user.local.checkpoints.')
    assert [entry['id'] for entry in alt_catalog_payload['entries']] == [response.json()['entry']['id']]



def test_model_api_add_model_endpoint_creates_huggingface_personal_catalog(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()
    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    response = client.post('/api/models/add', json={
        'source_provider': 'huggingface',
        'source_input': 'https://huggingface.co/acme/repo/blob/main/loras/demo_style.safetensors?download=true',
        'model_type': 'lora',
        'architecture': 'sdxl',
        'sub_architecture': 'illustrious',
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload['catalog']['catalog_id'] == 'user.personal.default'
    assert payload['catalog']['file_name'] == 'personal_download_catalog.catalog.json'
    assert payload['entry']['id'] == 'huggingface.lora.sdxl.illustrious.demo_style'
    assert payload['entry']['source_provider'] == 'huggingface'



def test_model_api_add_model_endpoint_rejects_invalid_civitai_input(tmp_path):
    catalog_dir = tmp_path / 'catalogs'
    catalog_dir.mkdir()

    manager = ModelManager(catalog_dirs=[catalog_dir], root_map={}).refresh()
    app = FastAPI()
    app.include_router(create_model_router(manager))
    client = TestClient(app)

    response = client.post('/api/models/add', json={
        'source_provider': 'civitai',
        'source_input': 'not-a-civitai-link',
        'model_type': 'checkpoint',
        'architecture': 'sdxl',
        'name': 'broken_model.safetensors',
    })
    assert response.status_code == 400
    assert 'CivitAI source' in response.json()['detail']
