from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import unquote, urlparse

from modules.model_loader import load_file_from_url

try:
    import requests
except ImportError:  # pragma: no cover - fallback path only matters if requests is missing.
    requests = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - fallback path only matters if huggingface_hub is missing.
    hf_hub_download = None

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

    if _is_huggingface_download_url(url):
        if partial_download:
            _cleanup_partial_download(destination)
        try:
            return _download_with_huggingface_hub(
                url=url,
                model_dir=model_dir,
                file_name=file_name,
                headers=headers,
            )
        except Exception as exc:
            print(f"Hugging Face Hub download failed for {url}: {exc}. Falling back to the Python downloader.")
            _cleanup_partial_download(destination)

        return load_file_from_url(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
            progress=progress,
            headers=_headers_with_default_user_agent(headers),
        )

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


@dataclass(frozen=True)
class _HuggingFaceURL:
    repo_id: str
    revision: str
    filename: str
    endpoint: str | None = None


def _headers_with_default_user_agent(headers: Iterable[tuple[str, str]] = ()) -> tuple[tuple[str, str], ...]:
    normalized = tuple(headers)
    if any(key.lower() == 'user-agent' for key, _value in normalized):
        return normalized
    return (*normalized, ('User-Agent', _ARIA2_USER_AGENT))


def _bearer_token_from_headers(headers: Iterable[tuple[str, str]] = ()) -> str | None:
    for key, value in headers:
        if key.lower() != 'authorization':
            continue
        prefix = 'Bearer '
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _parse_huggingface_url(url: str) -> _HuggingFaceURL | None:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    if not host:
        return None

    mirror = os.environ.get("HF_MIRROR", "").rstrip('/')
    mirror_host = urlparse(mirror).netloc.lower()
    if not (host.endswith('huggingface.co') or (mirror_host and host == mirror_host)):
        return None

    path_parts = [unquote(part) for part in parsed.path.split('/') if part]
    try:
        resolve_index = path_parts.index('resolve')
    except ValueError:
        return None
    if resolve_index < 1 or resolve_index + 2 >= len(path_parts):
        return None

    repo_id = '/'.join(path_parts[:resolve_index])
    revision = path_parts[resolve_index + 1]
    filename = '/'.join(path_parts[resolve_index + 2:])
    endpoint = mirror or None
    if host != 'huggingface.co' and not endpoint:
        endpoint = f'{parsed.scheme}://{parsed.netloc}'
    return _HuggingFaceURL(repo_id=repo_id, revision=revision, filename=filename, endpoint=endpoint)


def _download_with_huggingface_hub(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> str:
    if hf_hub_download is None:
        raise RuntimeError('huggingface_hub is not installed')

    spec = _parse_huggingface_url(url)
    if spec is None:
        raise ValueError(f'Unsupported Hugging Face download URL: {url}')

    destination = os.path.abspath(os.path.join(model_dir, file_name))
    os.makedirs(model_dir, exist_ok=True)

    request_headers = _headers_with_default_user_agent(headers)
    kwargs = {
        'repo_id': spec.repo_id,
        'filename': spec.filename,
        'revision': spec.revision,
        'local_dir': model_dir,
        'user_agent': _ARIA2_USER_AGENT,
        'headers': dict(request_headers),
    }
    token = _bearer_token_from_headers(request_headers)
    if token:
        kwargs['token'] = token
    if spec.endpoint:
        kwargs['endpoint'] = spec.endpoint

    try:
        downloaded_path = hf_hub_download(**kwargs)
    except TypeError:
        # Older huggingface_hub builds may not support the headers parameter.
        kwargs.pop('headers', None)
        downloaded_path = hf_hub_download(**kwargs)

    downloaded_path = os.path.abspath(downloaded_path)
    if downloaded_path != destination:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        # Same-filesystem rename avoids keeping a second model-sized copy in Colab.
        os.replace(downloaded_path, destination)
        _cleanup_empty_hf_download_dirs(downloaded_path, model_dir)
    return destination


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
    else:
        command = _build_generic_aria2_command(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
            headers=headers,
        )

    _run_aria2_command(command)
    return destination


def _cleanup_empty_hf_download_dirs(downloaded_path: str, model_dir: str) -> None:
    current = os.path.dirname(os.path.abspath(downloaded_path))
    stop = os.path.abspath(model_dir)
    while current != stop:
        try:
            if os.path.commonpath([current, stop]) != stop:
                break
        except ValueError:
            break
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)


def _cleanup_partial_download(destination: str) -> None:
    for candidate in (destination, f'{destination}.aria2'):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass
