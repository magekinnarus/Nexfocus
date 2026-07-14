from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable
from urllib.parse import urlparse

from modules.model_loader import load_file_from_url

try:
    import requests
except ImportError:  # pragma: no cover - fallback path only matters if requests is missing.
    requests = None

_ARIA2_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'


def download_file(
    url: str,
    *,
    model_dir: str,
    file_name: str | None = None,
    progress: bool = True,
    headers: Iterable[tuple[str, str]] = (),
    prefer_aria2: bool = True,
) -> str:
    if not file_name:
        file_name = os.path.basename(urlparse(url).path)
    else:
        file_name = file_name.replace('\\', '/')
        if '/' in file_name:
            sub_dir, file_name = file_name.rsplit('/', 1)
            model_dir = os.path.join(model_dir, sub_dir.replace('/', os.sep))

    os.makedirs(model_dir, exist_ok=True)

    destination = os.path.abspath(os.path.join(model_dir, file_name))
    partial_download = os.path.exists(f'{destination}.aria2')
    if os.path.exists(destination) and not partial_download:
        return destination

    if prefer_aria2 and shutil.which('aria2c'):
        try:
            return _download_with_aria2(
                url=url,
                model_dir=model_dir,
                file_name=file_name,
                headers=headers,
            )
        except Exception as exc:
            print(f"Aria2 download failed for {url}: {exc}. Falling back to the Python downloader.")
            _cleanup_partial_download(destination)

    if partial_download:
        _cleanup_partial_download(destination)

    return load_file_from_url(
        url=url,
        model_dir=model_dir,
        file_name=file_name,
        progress=progress,
        headers=headers,
    )


def _is_civitai_api_download_url(url: str) -> bool:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    return host.endswith('civitai.com') and '/api/download/models/' in path


def _resolve_civitai_direct_url(url: str, headers: Iterable[tuple[str, str]] = ()) -> str:
    # Use headers for resolution to ensure UA is set
    request_headers = {key: value for key, value in headers}
    if 'User-Agent' not in request_headers:
        request_headers['User-Agent'] = _ARIA2_USER_AGENT

    def _do_resolve(target_url):
        try:
            if requests is not None:
                # Try HEAD first as it's the most efficient
                try:
                    response = requests.head(target_url, headers=request_headers, allow_redirects=True, timeout=10)
                    response.raise_for_status()
                    return response.url
                except Exception as head_exc:
                    # Fallback to GET with stream=True if HEAD is forbidden or not allowed
                    status_code = getattr(getattr(head_exc, 'response', None), 'status_code', None)
                    if status_code in (403, 405) or "403" in str(head_exc) or "405" in str(head_exc):
                        with requests.get(target_url, headers=request_headers, allow_redirects=True, timeout=10, stream=True) as response:
                            response.raise_for_status()
                            return response.url
                    raise head_exc

            import urllib.request
            request = urllib.request.Request(target_url, headers=request_headers, method='HEAD')
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.geturl()
            except Exception:
                # Basic urllib fallback to GET
                request.method = 'GET'
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.geturl()
        except Exception as e:
            # Fallback to .red if .com fails for CivitAI URLs
            parsed = urlparse(target_url)
            if parsed.netloc.lower() == 'civitai.com':
                red_url = target_url.replace('civitai.com', 'civitai.red', 1)
                print(f"CivitAI .com resolution failed for {target_url}: {e}. Retrying via .red ...")
                return _do_resolve(red_url)
            raise e

    return _do_resolve(url)


def _is_huggingface_download_url(url: str) -> bool:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    mirror_host = urlparse(os.environ.get("HF_MIRROR", "")).netloc.lower()
    return host.endswith('huggingface.co') or (mirror_host and host == mirror_host)


def _resolve_hf_direct_url(url: str, headers: Iterable[tuple[str, str]] = ()) -> str:
    # Use headers for resolution to ensure UA is set
    request_headers = {key: value for key, value in headers}
    if 'User-Agent' not in request_headers:
        request_headers['User-Agent'] = _ARIA2_USER_AGENT

    try:
        if requests is not None:
            # Hugging Face Xet signed endpoints can reject HEAD even when GET is
            # publicly authorized. Stream the GET so only response headers are read.
            with requests.get(url, headers=request_headers, allow_redirects=True, timeout=15, stream=True) as response:
                response.raise_for_status()
                return response.url

        import urllib.request
        request = urllib.request.Request(url, headers=request_headers, method='GET')
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.geturl()
    except Exception as e:
        print(f"Hugging Face redirect resolution failed for {url}: {e}. Downloading directly.")
        return url


def _build_generic_aria2_command(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> list[str]:
    command = [
        'aria2c',
        '--console-log-level=warn',
        '--check-certificate=false',
        '--retry-wait=2',
        '-c',
        '-x', '8',
        '-s', '8',
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
    ]

    for key, value in headers:
        command.extend(['--header', f'{key}: {value}'])

    command.append(url)
    return command


def _build_civitai_aria2_command(*, direct_url: str, model_dir: str, file_name: str) -> list[str]:
    return [
        'aria2c',
        '--console-log-level=warn',
        f'--user-agent={_ARIA2_USER_AGENT}',
        '--check-certificate=false',
        '-x', '16',
        '-s', '16',
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
        direct_url,
    ]


def _build_hf_aria2_command(
    *,
    direct_url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> list[str]:
    command = [
        'aria2c',
        '--console-log-level=warn',
        f'--user-agent={_ARIA2_USER_AGENT}',
        '--check-certificate=false',
        '--retry-wait=2',
        '-c',
        '-x', '8',
        '-s', '8',
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
    ]
    for key, value in headers:
        if key.lower() != 'user-agent':
            command.extend(['--header', f'{key}: {value}'])
        else:
            command[2] = f'--user-agent={value}'
    command.append(direct_url)
    return command


def _run_aria2_command(command: list[str]) -> None:
    subprocess.check_call(command)


def _download_with_aria2(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> str:
    destination = os.path.abspath(os.path.join(model_dir, file_name))
    headers = tuple(headers)

    if _is_civitai_api_download_url(url):
        direct_url = _resolve_civitai_direct_url(url, headers=headers)
        _cleanup_partial_download(destination)
        command = _build_civitai_aria2_command(
            direct_url=direct_url,
            model_dir=model_dir,
            file_name=file_name,
        )
    elif _is_huggingface_download_url(url):
        direct_url = _resolve_hf_direct_url(url, headers=headers)
        command = _build_hf_aria2_command(
            direct_url=direct_url,
            model_dir=model_dir,
            file_name=file_name,
            headers=headers,
        )
    else:
        command = _build_generic_aria2_command(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
            headers=headers,
        )

    _run_aria2_command(command)
    return destination



def _cleanup_partial_download(destination: str) -> None:
    for candidate in (destination, f'{destination}.aria2'):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass
