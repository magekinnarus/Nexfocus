from pathlib import Path

from modules.model_download.runtime import (
    _build_civitai_aria2_command,
    _download_with_huggingface_hub,
    _headers_with_default_user_agent,
    _parse_huggingface_url,
    download_file,
)


def test_runtime_resumes_aria2_partial_when_destination_exists(tmp_path, monkeypatch):
    destination = tmp_path / 'model.safetensors'
    destination.write_bytes(b'partial')
    Path(f'{destination}.aria2').write_bytes(b'control')
    calls = []

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')

    def fake_download_with_aria2(**kwargs):
        calls.append(kwargs)
        Path(f'{destination}.aria2').unlink()
        destination.write_bytes(b'complete')
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
    assert calls[0]['url'].startswith('https://example.com/')


def test_runtime_uses_huggingface_hub_for_hf_urls(tmp_path, monkeypatch):
    calls = []
    destination = tmp_path / 'model.safetensors'

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')
    monkeypatch.setattr(
        'modules.model_download.runtime._download_with_aria2',
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError('HF must not use Aria2')),
    )

    def fake_hf_download(**kwargs):
        calls.append(kwargs)
        destination.write_bytes(b'complete')
        return str(destination)

    monkeypatch.setattr('modules.model_download.runtime._download_with_huggingface_hub', fake_hf_download)

    result = download_file(
        url='https://huggingface.co/example/model/resolve/main/model.safetensors',
        model_dir=str(tmp_path),
        file_name=destination.name,
        prefer_aria2=True,
    )

    assert result == str(destination)
    assert len(calls) == 1
    assert calls[0]['url'].startswith('https://huggingface.co/')


def test_civitai_uses_browser_user_agent_and_parallel_connections():
    command = _build_civitai_aria2_command(
        direct_url='https://cdn.example/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
    )

    assert any(item.startswith('--user-agent=Mozilla/5.0') for item in command)
    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'


def test_hf_url_parser_extracts_repo_revision_and_filename():
    parsed = _parse_huggingface_url(
        'https://huggingface.co/Old-Fisherman/Fooocus_Nex/resolve/main/checkpoints/sdxl/beretMixReal_v110.safetensors'
    )

    assert parsed.repo_id == 'Old-Fisherman/Fooocus_Nex'
    assert parsed.revision == 'main'
    assert parsed.filename == 'checkpoints/sdxl/beretMixReal_v110.safetensors'


def test_hf_download_uses_local_dir_and_moves_repo_subpath(tmp_path, monkeypatch):
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        downloaded = Path(kwargs['local_dir']) / kwargs['filename']
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b'complete')
        return str(downloaded)

    monkeypatch.setattr('modules.model_download.runtime.hf_hub_download', fake_hf_hub_download)

    result = _download_with_huggingface_hub(
        url='https://huggingface.co/example/model/resolve/main/subdir/model.safetensors',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
        headers=iter((('Authorization', 'Bearer token'),)),
    )

    assert result == str(tmp_path / 'model.safetensors')
    assert Path(result).read_bytes() == b'complete'
    assert not (tmp_path / 'subdir' / 'model.safetensors').exists()
    assert calls[0]['local_dir'] == str(tmp_path)
    assert calls[0]['filename'] == 'subdir/model.safetensors'
    assert calls[0]['token'] == 'token'
    assert calls[0]['headers']['User-Agent'].startswith('Mozilla/5.0')


def test_hf_default_user_agent_preserves_custom_header():
    headers = _headers_with_default_user_agent((('User-Agent', 'CustomUA'),))

    assert headers == (('User-Agent', 'CustomUA'),)
