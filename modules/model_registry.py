import json
import os
import zipfile
from functools import lru_cache

from modules.model_download.runtime import download_file, validate_downloaded_file


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assets = data.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError(f"Invalid asset manifest: {path}")
    return assets


@lru_cache(maxsize=1)
def _load_asset_index():
    import modules.config as config

    asset_index = {}
    manifest_dir = config.get_download_manifest_asset_dir()
    if not os.path.isdir(manifest_dir):
        return asset_index

    for name in sorted(os.listdir(manifest_dir)):
        if not name.lower().endswith(".json"):
            continue
        manifest_path = os.path.join(manifest_dir, name)
        for asset in _load_manifest(manifest_path):
            asset_id = asset.get("id")
            if not asset_id:
                raise ValueError(f"Asset manifest entry in {manifest_path} is missing id")
            if asset_id in asset_index:
                raise ValueError(f"Duplicate asset id found: {asset_id}")
            asset_index[asset_id] = asset
    return asset_index


def clear_asset_index_cache():
    _load_asset_index.cache_clear()


def get_asset(asset_id):
    return _load_asset_index().get(asset_id)


def _asset_matches(asset, filters):
    for key, expected in filters.items():
        if expected is None:
            continue
        if asset.get(key) != expected:
            return False
    return True


def list_assets(channel=None, method=None, kind=None, category=None, engine_family=None, internal_only=None, destination=None):
    filters = {
        "channel": channel,
        "method": method,
        "kind": kind,
        "category": category,
        "engine_family": engine_family,
        "internal_only": internal_only,
        "destination": destination,
    }

    results = []
    for asset in _load_asset_index().values():
        if _asset_matches(asset, filters):
            results.append(asset)
    return results


def list_asset_ids(**filters):
    return [asset["id"] for asset in list_assets(**filters)]


def _require_asset(asset_id):
    asset = get_asset(asset_id)
    if asset is None:
        raise KeyError(f"Unknown asset id: {asset_id}")
    return asset


def resolve_asset_root(asset):
    import modules.config as config

    destination = asset.get("destination")
    if not destination:
        raise ValueError(f"Asset {asset.get('id')} is missing destination")
    return config.get_preferred_asset_root_path(
        destination,
        file_name=asset.get("archive_name") or os.path.basename(str(asset.get("relative_path") or "")).strip(),
        relative_path=asset.get("relative_path"),
    )


def resolve_asset_path(asset_id):
    asset = _require_asset(asset_id)
    artifact_type = asset.get("artifact_type", "file")
    root = resolve_asset_root(asset)

    if artifact_type == "archive":
        extract_subdir = asset.get("extract_subdir") or asset["id"].replace(".", "_")
        return os.path.join(root, extract_subdir)

    relative_path = asset.get("relative_path")
    if not relative_path:
        raise ValueError(f"File asset {asset_id} is missing relative_path")
    return os.path.join(root, relative_path)


def _get_asset_sources(asset):
    return [source for source in asset.get("sources", []) if source.get("url")]


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _validate_file_asset(asset, path):
    if not validate_downloaded_file(path):
        return False
    expected_size = asset.get('expected_size_bytes')
    if expected_size is not None:
        try:
            return os.path.getsize(path) == int(expected_size)
        except (OSError, TypeError, ValueError):
            return False
    return True


def _verify_expected_files(base_dir, expected_files):
    if not expected_files:
        return True
    return all(os.path.exists(os.path.join(base_dir, rel_path)) for rel_path in expected_files)


def asset_exists(asset_id):
    asset = _require_asset(asset_id)
    artifact_type = asset.get("artifact_type", "file")
    target_path = resolve_asset_path(asset_id)
    if artifact_type == "archive":
        return _verify_expected_files(target_path, asset.get("expected_files", []))
    return os.path.exists(target_path)


def ensure_asset(asset_id, progress=True):
    asset = _require_asset(asset_id)
    artifact_type = asset.get("artifact_type", "file")
    target_path = resolve_asset_path(asset_id)

    if artifact_type == "archive":
        return _ensure_archive_asset(asset, target_path, progress=progress)
    return _ensure_file_asset(asset, target_path, progress=progress)


def _ensure_file_asset(asset, target_path, progress=True):
    has_aria2_state = os.path.exists(f'{target_path}.aria2')
    if os.path.exists(target_path) and not has_aria2_state:
        if _validate_file_asset(asset, target_path):
            return target_path
        print(f"Discarding invalid cached asset {asset['id']}: {target_path}")
        try:
            os.remove(target_path)
        except OSError:
            pass

    sources = _get_asset_sources(asset)
    if not sources:
        raise FileNotFoundError(f"Asset {asset['id']} is missing and has no download source")

    _ensure_parent_dir(target_path)
    download_dir = os.path.dirname(target_path) or resolve_asset_root(asset)
    file_name = os.path.basename(target_path)

    last_error = None
    for source in sources:
        try:
            downloaded_path = download_file(
                url=source["url"],
                model_dir=download_dir,
                file_name=file_name,
                progress=progress,
                headers=source.get("headers", ()),
            )
            if not _validate_file_asset(asset, downloaded_path):
                try:
                    os.remove(downloaded_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Downloaded asset {asset['id']} failed its format or expected-size check"
                )
            return downloaded_path
        except Exception as exc:
            last_error = exc
            print(f"Failed to download asset {asset['id']} from {source['url']}: {exc}")

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Asset {asset['id']} is missing and has no download source")


def _ensure_archive_asset(asset, extract_dir, progress=True):
    expected_files = asset.get("expected_files", [])
    if _verify_expected_files(extract_dir, expected_files):
        return extract_dir

    sources = _get_asset_sources(asset)
    if not sources:
        raise FileNotFoundError(f"Archive asset {asset['id']} is missing and has no download source")

    archive_name = asset.get("archive_name")
    if not archive_name:
        raise ValueError(f"Archive asset {asset['id']} is missing archive_name")

    archive_cache_dir = os.path.join(resolve_asset_root(asset), "_archives")
    os.makedirs(archive_cache_dir, exist_ok=True)

    last_error = None
    archive_path = None
    for source in sources:
        try:
            archive_path = download_file(
                url=source["url"],
                model_dir=archive_cache_dir,
                file_name=archive_name,
                progress=progress,
                headers=source.get("headers", ()),
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"Failed to download archive asset {asset['id']} from {source['url']}: {exc}")

    if archive_path is None:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"Archive asset {asset['id']} is missing and has no download source")

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    if asset.get("flatten_first_subdir", False):
        _flatten_first_subdir(extract_dir)

    if not _verify_expected_files(extract_dir, expected_files):
        raise FileNotFoundError(
            f"Archive asset {asset['id']} did not extract the expected files into {extract_dir}"
        )
    return extract_dir


def _flatten_first_subdir(directory):
    entries = [name for name in os.listdir(directory) if name != "__MACOSX"]
    if len(entries) != 1:
        return

    nested_dir = os.path.join(directory, entries[0])
    if not os.path.isdir(nested_dir):
        return

    for name in os.listdir(nested_dir):
        src = os.path.join(nested_dir, name)
        dst = os.path.join(directory, name)
        if os.path.exists(dst):
            continue
        os.replace(src, dst)

    try:
        os.rmdir(nested_dir)
    except OSError:
        pass

