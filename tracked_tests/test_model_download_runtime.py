from pathlib import Path

from modules.model_download.runtime import (
    _build_civitai_aria2_command,
    _build_hf_aria2_command,
    _download_with_aria2,
    _resolve_hf_direct_url,
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
        url='https://huggingface.co/example/model.safetensors',
        model_dir=str(tmp_path),
        file_name=destination.name,
        prefer_aria2=True,
    )

    assert result == str(destination)
    assert len(calls) == 1
    assert calls[0]['url'].startswith('https://huggingface.co/')


def test_hf_redirect_resolution_uses_streamed_get(monkeypatch):
    calls = []

    class FakeResponse:
        url = 'https://cdn.example/model.safetensors'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr('modules.model_download.runtime.requests.get', fake_get)

    resolved = _resolve_hf_direct_url('https://huggingface.co/example/model/resolve/main/model.safetensors')

    assert resolved == 'https://cdn.example/model.safetensors'
    assert calls[0][1]['stream'] is True
    assert calls[0][1]['allow_redirects'] is True


def test_hf_aria2_command_uses_user_agent_and_parallel_connections():
    command = _build_hf_aria2_command(
        direct_url='https://cdn.example/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
        headers=(('Authorization', 'Bearer token'),),
    )

    assert any(item.startswith('--user-agent=') for item in command)
    assert '--check-certificate=false' in command
    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'
    assert ['--header', 'Authorization: Bearer token'] == command[
        command.index('--header'):command.index('--header') + 2
    ]


def test_civitai_and_hf_commands_share_parallel_browser_posture():
    command = _build_civitai_aria2_command(
        direct_url='https://cdn.example/model.safetensors',
        model_dir='/models',
        file_name='model.safetensors',
    )

    assert any(item.startswith('--user-agent=Mozilla/5.0') for item in command)
    assert command[command.index('-x') + 1] == '16'
    assert command[command.index('-s') + 1] == '16'


def test_hf_download_reuses_generator_headers_for_aria2(tmp_path, monkeypatch):
    commands = []

    def fake_resolve(url, headers):
        assert dict(headers)['Authorization'] == 'Bearer token'
        return 'https://cdn.example/model.safetensors'

    monkeypatch.setattr('modules.model_download.runtime._resolve_hf_direct_url', fake_resolve)
    monkeypatch.setattr('modules.model_download.runtime._run_aria2_command', commands.append)

    _download_with_aria2(
        url='https://huggingface.co/example/model/resolve/main/model.safetensors',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
        headers=iter((('Authorization', 'Bearer token'),)),
    )

    command = commands[0]
    assert ['--header', 'Authorization: Bearer token'] == command[
        command.index('--header'):command.index('--header') + 2
    ]
