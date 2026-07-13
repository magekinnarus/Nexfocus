from pathlib import Path

from modules.model_download.runtime import download_file


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
