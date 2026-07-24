import json
import sys

sys.argv = [sys.argv[0]]

from pathlib import Path

from backend.auxiliary_workers import background_removal_worker, mat_inpaint_worker
from modules import config, setup_utils, flags


class _DummyRemover:
    def __init__(self, jit, ckpt):
        self.jit = jit
        self.ckpt = ckpt


class _DummyMat:
    def __init__(self):
        self.state_dict = None
        self.eval_called = False
        self.dtype = None

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def eval(self):
        self.eval_called = True

    def to(self, dtype):
        self.dtype = dtype


def test_downloading_inpaint_models_uses_manifest(monkeypatch):
    calls = []

    def fake_ensure_asset(asset_id, progress=True):
        calls.append((asset_id, progress))
        return f'/tmp/{asset_id}'

    monkeypatch.setattr('modules.model_registry.ensure_asset', fake_ensure_asset)

    result = config.downloading_inpaint_models('v2.6')

    assert result == '/tmp/inpaint.fooocus_patch.v2_6'
    assert calls == [('inpaint.fooocus_patch.v2_6', True)]


def test_downloading_inpaint_models_normalizes_legacy_versions(monkeypatch):
    calls = []

    def fake_ensure_asset(asset_id, progress=True):
        calls.append((asset_id, progress))
        return f'/tmp/{asset_id}'

    monkeypatch.setattr('modules.model_registry.ensure_asset', fake_ensure_asset)

    result = config.downloading_inpaint_models('v1')

    assert result == '/tmp/inpaint.fooocus_patch.v2_6'
    assert calls == [('inpaint.fooocus_patch.v2_6', True)]


def test_normalize_inpaint_engine_version_collapses_legacy_versions():
    assert flags.normalize_inpaint_engine_version('v1') == flags.INPAINT_ENGINE_V26
    assert flags.normalize_inpaint_engine_version('v2.5') == flags.INPAINT_ENGINE_V26
    assert flags.normalize_inpaint_engine_version('v2.6') == flags.INPAINT_ENGINE_V26
    assert flags.normalize_inpaint_engine_version('None') == flags.INPAINT_ENGINE_NONE


def test_download_models_prefetches_manifest_upscalers(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'dummy.safetensors').write_text('x', encoding='utf-8')

    monkeypatch.setattr(config, 'paths_checkpoints', [str(checkpoint_dir)])
    monkeypatch.setattr(config, 'path_vae_approx', str(tmp_path / 'vae_approx'))
    monkeypatch.setattr(config, 'path_embeddings', str(tmp_path / 'embeddings'))
    monkeypatch.setattr(config, 'paths_loras', [str(tmp_path / 'loras')])
    monkeypatch.setattr(config, 'path_vae', str(tmp_path / 'vae'))
    monkeypatch.setattr(config, 'path_upscale_models', [str(tmp_path / 'upscale_models')])

    download_calls = []
    ensure_calls = []

    monkeypatch.setattr(setup_utils, 'download_file', lambda **kwargs: download_calls.append(kwargs) or str(Path(kwargs['model_dir']) / kwargs['file_name']))
    monkeypatch.setattr('modules.model_registry.list_assets', lambda **kwargs: [
        {'id': 'upscale.a'},
        {'id': 'upscale.b'},
    ] if kwargs.get('category') == 'upscale' and kwargs.get('internal_only') is True else [])
    monkeypatch.setattr('modules.model_registry.ensure_asset', lambda asset_id, progress=False: ensure_calls.append((asset_id, progress)) or f'/tmp/{asset_id}')

    _, _, downloaded_assets = setup_utils.download_models('base', {}, {}, {}, {}, {})

    assert ('upscale.a', False) in ensure_calls
    assert ('upscale.b', False) in ensure_calls
    assert len(download_calls) == len(setup_utils.vae_approx_filenames)
    assert any(call['file_name'] == 'taesdxl_decoder.pth' for call in download_calls)
    assert any(call['file_name'] == 'taef1_decoder.pth' for call in download_calls)
    assert downloaded_assets is False


def test_download_preset_models_skips_startup_support_noise(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()

    monkeypatch.setattr(config, 'paths_checkpoints', [str(checkpoint_dir)])
    monkeypatch.setattr(config, 'path_vae_approx', str(tmp_path / 'vae_approx'))
    monkeypatch.setattr(config, 'path_embeddings', str(tmp_path / 'embeddings'))
    monkeypatch.setattr(config, 'paths_loras', [str(tmp_path / 'loras')])
    monkeypatch.setattr(config, 'path_vae', str(tmp_path / 'vae'))
    monkeypatch.setattr(config, 'path_upscale_models', [str(tmp_path / 'upscale_models')])

    download_calls = []
    ensure_calls = []

    def fake_download_file(**kwargs):
        download_calls.append(kwargs)
        target_path = Path(kwargs['model_dir']) / kwargs['file_name']
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text('x', encoding='utf-8')
        return str(target_path)

    monkeypatch.setattr(setup_utils, 'download_file', fake_download_file)
    monkeypatch.setattr('modules.model_registry.ensure_asset', lambda asset_id, progress=False: ensure_calls.append((asset_id, progress)) or f'/tmp/{asset_id}')

    _, _, downloaded_assets = setup_utils.download_preset_models(
        'base',
        {},
        {},
        {
            'sdxl_special_lora.safetensors': 'https://example.invalid/special.safetensors',
        },
        {},
        {},
    )

    assert downloaded_assets is True
    assert [call['file_name'] for call in download_calls] == ['sdxl_special_lora.safetensors']
    assert ensure_calls == []


def test_background_removal_worker_load_uses_manifest(monkeypatch):
    calls = []

    monkeypatch.setattr(background_removal_worker.model_registry, 'ensure_asset', lambda asset_id, progress=True: calls.append((asset_id, progress)) or '/tmp/ckpt_base.pth')
    monkeypatch.setattr(background_removal_worker, 'Remover', _DummyRemover)

    worker = background_removal_worker.BackgroundRemovalWorker()
    worker.load(jit=False)

    try:
        assert calls == [('removals.background.inspyrenet.base', True)]
        assert isinstance(worker.remover, _DummyRemover)
        assert worker.remover.ckpt == '/tmp/ckpt_base.pth'
        assert worker.remover.jit is False
    finally:
        worker.teardown()


def test_mat_inpaint_worker_load_uses_manifest(monkeypatch):
    calls = []

    monkeypatch.setattr(mat_inpaint_worker.model_registry, 'ensure_asset', lambda asset_id, progress=True: calls.append((asset_id, progress)) or '/tmp/Places_512_FullData_G.pth')
    monkeypatch.setattr(mat_inpaint_worker.torch, 'load', lambda *args, **kwargs: {'synthesis.block.weight': 1, 'mapping.fc.weight': 2})
    monkeypatch.setattr(mat_inpaint_worker, 'MAT', _DummyMat)

    worker = mat_inpaint_worker.MatInpaintWorker()
    worker.load()

    try:
        assert calls == [('removals.object.mat.places512', True)]
        assert isinstance(worker.model, _DummyMat)
        assert worker.model.eval_called is True
        assert 'model.synthesis.block.weight' in worker.model.state_dict
        assert 'model.mapping.fc.weight' in worker.model.state_dict
    finally:
        worker.teardown()


def test_flux_fp16_t5_manifest_is_civitai_only_with_exact_size():
    manifest_path = Path('configs/download_manifests/assets/assets_flux_fill.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    asset = next(
        item
        for item in manifest['assets']
        if item['id'] == 'inpaint.flux_fill.text_encoder.t5xxl.fp16'
    )

    assert asset['expected_size_bytes'] == 9787841024
    assert [source['url'] for source in asset['sources']] == [
        'https://civitai.com/api/download/models/787954'
    ]
