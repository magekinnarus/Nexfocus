from pathlib import Path

from modules.model_download.runtime import (
    _build_civitai_aria2_command,
    _build_github_aria2_command,
    _build_generic_aria2_command,
    _download_with_aria2,
    _with_download_query,
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
        destination.write_bytes(b'complete')
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
    assert destination.read_bytes() == b'complete'
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
