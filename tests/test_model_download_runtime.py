import json
from pathlib import Path

import pytest

from modules.model_download.runtime import (
    _CivitAIAuthenticationError,
    _build_civitai_aria2_command,
    _build_github_aria2_command,
    _build_generic_aria2_command,
    _download_with_aria2,
    _validate_civitai_direct_response,
    _with_civitai_token,
    _with_download_query,
    download_file,
    validate_downloaded_file,
)


def _write_minimal_safetensors(path: Path) -> None:
    header = b'{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    path.write_bytes(len(header).to_bytes(8, byteorder='little') + header + b'\x00')


def test_runtime_resumes_aria2_partial_when_destination_exists(tmp_path, monkeypatch):
    destination = tmp_path / 'model.safetensors'
    destination.write_bytes(b'partial')
    Path(f'{destination}.aria2').write_bytes(b'control')
    calls = []

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')

    def fake_download_with_aria2(**kwargs):
        calls.append(kwargs)
        Path(f'{destination}.aria2').unlink()
        _write_minimal_safetensors(destination)
        return str(destination)

    monkeypatch.setattr('modules.model_download.runtime._download_with_aria2', fake_download_with_aria2)

    result = download_file(
        url='https://example.com/model.safetensors',
        model_dir=str(tmp_path),
        file_name=destination.name,
        prefer_aria2=True,
    )

    assert result == str(destination)
    assert len(calls) == 1
    assert calls[0]['url'] == 'https://example.com/model.safetensors'


def test_hf_download_bypasses_aria2_and_cleans_partial_state(tmp_path, monkeypatch):
    destination = tmp_path / 'model.safetensors'
    destination.write_bytes(b'partial')
    Path(f'{destination}.aria2').write_bytes(b'control')

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')
    monkeypatch.setattr(
        'modules.model_download.runtime._download_with_aria2',
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError('HF must bypass Aria2')),
    )

    calls = {}

    def fake_load_file_from_url(**kwargs):
        calls.update(kwargs)
        _write_minimal_safetensors(destination)
        return str(destination)

    monkeypatch.setattr(
        'modules.model_download.runtime.load_file_from_url',
        fake_load_file_from_url,
    )

    result = download_file(
        url='https://huggingface.co/example/model/resolve/main/model.safetensors',
        model_dir=str(tmp_path),
        file_name=destination.name,
        prefer_aria2=True,
    )

    assert result == str(destination)
    assert validate_downloaded_file(str(destination)) is True
    assert not Path(f'{destination}.aria2').exists()
    assert calls['url'].endswith('?download=true')


def test_generic_aria2_uses_four_connections():
    command = _build_generic_aria2_command(
        url='https://cdn.example/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
        headers=(('Authorization', 'Bearer token'),),
    )

    assert command[command.index('-x') + 1] == '4'
    assert command[command.index('-s') + 1] == '4'
    assert '--max-tries=20' in command
    assert '--retry-wait=5' in command
    assert '--file-allocation=none' in command
    assert ['--header', 'Authorization: Bearer token'] == command[
        command.index('--header'):command.index('--header') + 2
    ]


def test_hf_download_query_is_added_without_losing_existing_params():
    assert _with_download_query(
        'https://huggingface.co/example/model/resolve/main/model.safetensors'
    ) == 'https://huggingface.co/example/model/resolve/main/model.safetensors?download=true'
    assert _with_download_query(
        'https://huggingface.co/example/model/resolve/main/model.safetensors?foo=bar'
    ) == 'https://huggingface.co/example/model/resolve/main/model.safetensors?foo=bar&download=true'
    assert _with_download_query(
        'https://huggingface.co/example/model/resolve/main/model.safetensors?download=false'
    ) == 'https://huggingface.co/example/model/resolve/main/model.safetensors?download=true'


def test_github_aria2_uses_sixteen_connections():
    command = _build_github_aria2_command(
        url='https://github.com/example/repo/releases/download/test/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
        headers=(),
    )

    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'
    assert '--max-tries=20' in command
    assert '--retry-wait=5' in command
    assert '--file-allocation=none' in command


def test_civitai_aria2_uses_browser_user_agent_and_sixteen_connections():
    command = _build_civitai_aria2_command(
        direct_url='https://cdn.example/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
    )

    assert any(item.startswith('--user-agent=Mozilla/5.0') for item in command)
    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'
    assert '--max-tries=20' in command
    assert '--retry-wait=5' in command
    assert '--file-allocation=none' in command


def test_github_download_uses_sixteen_connection_aria2(tmp_path, monkeypatch):
    commands = []

    monkeypatch.setattr('modules.model_download.runtime._run_aria2_command', commands.append)

    _download_with_aria2(
        url='https://github.com/example/repo/releases/download/test/model.safetensors',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
    )

    command = commands[0]
    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'


def test_runtime_rejects_html_aria2_payload_and_uses_python_fallback(tmp_path, monkeypatch):
    destination = tmp_path / 'model.safetensors'
    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')

    def fake_download_with_aria2(**_kwargs):
        destination.write_bytes(b'<Error>signed URL expired</Error>')
        return str(destination)

    def fake_load_file_from_url(**_kwargs):
        _write_minimal_safetensors(destination)
        return str(destination)

    monkeypatch.setattr(
        'modules.model_download.runtime._download_with_aria2',
        fake_download_with_aria2,
    )
    monkeypatch.setattr(
        'modules.model_download.runtime.load_file_from_url',
        fake_load_file_from_url,
    )

    result = download_file(
        url='https://civitai.com/api/download/models/12345',
        model_dir=str(tmp_path),
        file_name=destination.name,
    )

    assert result == str(destination)
    assert validate_downloaded_file(result) is True


def test_runtime_discards_invalid_cached_safetensors(tmp_path, monkeypatch):
    destination = tmp_path / 'model.safetensors'
    destination.write_bytes(b'<html>not a model</html>')
    calls = []

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: None)

    def fake_load_file_from_url(**kwargs):
        calls.append(kwargs)
        assert not destination.exists()
        _write_minimal_safetensors(destination)
        return str(destination)

    monkeypatch.setattr(
        'modules.model_download.runtime.load_file_from_url',
        fake_load_file_from_url,
    )

    result = download_file(
        url='https://example.com/model.safetensors',
        model_dir=str(tmp_path),
        file_name=destination.name,
    )

    assert result == str(destination)
    assert len(calls) == 1
    assert validate_downloaded_file(result) is True


def test_registry_discards_invalid_cache_and_advances_to_fallback_source(tmp_path, monkeypatch):
    from modules import model_registry

    destination = tmp_path / 'model.safetensors'
    destination.write_bytes(b'<Error>expired signed URL</Error>')
    calls = []
    asset = {
        'id': 'test.model',
        'sources': [
            {'url': 'https://primary.example/model.safetensors'},
            {'url': 'https://fallback.example/model.safetensors'},
        ],
    }

    def fake_download_file(*, url, **_kwargs):
        calls.append(url)
        assert not destination.exists()
        if 'primary.example' in url:
            raise RuntimeError('Primary source returned an invalid payload')
        _write_minimal_safetensors(destination)
        return str(destination)

    monkeypatch.setattr(model_registry, 'download_file', fake_download_file)

    result = model_registry._ensure_file_asset(asset, str(destination), progress=False)

    assert result == str(destination)
    assert calls == [source['url'] for source in asset['sources']]
    assert validate_downloaded_file(result) is True


def test_civitai_token_is_added_to_internal_api_download(monkeypatch):
    monkeypatch.setenv('CIVITAI_TOKEN', 'secret-token')

    resolved = _with_civitai_token('https://civitai.com/api/download/models/787954')

    assert resolved == 'https://civitai.com/api/download/models/787954?token=secret-token'


def test_civitai_login_or_html_response_fails_closed():
    with pytest.raises(RuntimeError, match='CIVITAI_TOKEN'):
        _validate_civitai_direct_response(
            'https://auth.civitai.com/login?reason=download-auth',
            content_type='text/html; charset=UTF-8',
        )


def test_civitai_auth_failure_does_not_fall_back_to_python(tmp_path, monkeypatch):
    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')
    monkeypatch.setattr(
        'modules.model_download.runtime._download_with_aria2',
        lambda **_kwargs: (_ for _ in ()).throw(
            _CivitAIAuthenticationError('Set CIVITAI_TOKEN and retry')
        ),
    )
    monkeypatch.setattr(
        'modules.model_download.runtime.load_file_from_url',
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError('Authentication failures must not use Python fallback')
        ),
    )

    with pytest.raises(RuntimeError, match='CIVITAI_TOKEN'):
        download_file(
            url='https://civitai.com/api/download/models/787954',
            model_dir=str(tmp_path),
            file_name='t5xxl_fp16.safetensors',
        )


def test_flux_fp16_t5_manifest_is_civitai_only_with_exact_size():
    manifest = json.loads(
        Path('configs/download_manifests/assets/assets_flux_fill.json').read_text(encoding='utf-8')
    )
    asset = next(
        item
        for item in manifest['assets']
        if item['id'] == 'inpaint.flux_fill.text_encoder.t5xxl.fp16'
    )

    assert asset['expected_size_bytes'] == 9787841024
    assert [source['url'] for source in asset['sources']] == [
        'https://civitai.com/api/download/models/787954'
    ]
