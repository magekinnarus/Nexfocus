import os
import sys
import time
from unittest.mock import patch

import torch

# Ensure we are in the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.argv = [sys.argv[0]]

import ldm_patched.modules.sd1_clip as sd1_clip
import modules.config as config

def test_expand_directory_list_caching():
    dummy_dir = os.path.abspath("test_cache_dir")
    os.makedirs(dummy_dir, exist_ok=True)
    
    # 1. First call should use os.walk
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(dummy_dir, [], [])]
        
        start_time = time.time()
        res1 = sd1_clip.expand_directory_list([dummy_dir])
        print(f"First call took: {(time.time() - start_time)*1000:.2f}ms")
        
        assert mock_walk.called
        assert dummy_dir in res1
        
        # 2. Second call should hit the cache (zero os.walk calls)
        mock_walk.reset_mock()
        start_time = time.time()
        res2 = sd1_clip.expand_directory_list([dummy_dir])
        print(f"Second call (cached) took: {(time.time() - start_time)*1000:.2f}ms")
        
        assert not mock_walk.called
        assert res1 == res2
        
    # 3. Wait for TTL (5s) and check if it re-walks
    print("Waiting 5.1s for TTL expiry...")
    time.sleep(5.1)
    
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(dummy_dir, [], [])]
        sd1_clip.expand_directory_list([dummy_dir])
        assert mock_walk.called
        print("Re-scanned after TTL expiry successfully.")

    os.rmdir(dummy_dir)
    print("Test passed!")


def test_resolve_embedding_path_uses_cached_lookup(tmp_path, monkeypatch):
    embed_dir = tmp_path / "embeddings"
    nested_dir = embed_dir / "nested"
    nested_dir.mkdir(parents=True)
    embed_file = nested_dir / "slow-neg.pt"
    embed_file.write_bytes(b"placeholder")

    monkeypatch.setattr(config, "path_embeddings", str(embed_dir))
    monkeypatch.setitem(config.asset_root_path_groups, "embeddings", [str(embed_dir)])
    monkeypatch.setattr(config, "embedding_filenames", ["nested/slow-neg.pt"])

    config.rebuild_embedding_path_lookup()

    assert config.resolve_embedding_path("slow-neg") == str(embed_file.resolve())
    assert config.resolve_embedding_path("slow-neg.pt") == str(embed_file.resolve())
    assert config.resolve_embedding_path("nested/slow-neg") == str(embed_file.resolve())


def test_load_embed_caches_resolved_file(tmp_path):
    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir(parents=True)
    embed_file = embed_dir / "cached-neg.pt"
    embed_file.write_bytes(b"placeholder")

    sd1_clip._embedding_file_cache.clear()

    fake_tensor = torch.ones((1, 768))
    load_calls = {"count": 0}

    def fake_torch_load(*args, **kwargs):
        load_calls["count"] += 1
        return {"clip_l": fake_tensor}

    with patch.object(sd1_clip, "expand_directory_list", return_value=[str(embed_dir)]), \
         patch.object(sd1_clip.torch, "load", new=fake_torch_load):
        first = sd1_clip.load_embed("cached-neg", [str(embed_dir)], 768, embed_key="clip_l")
        second = sd1_clip.load_embed("cached-neg", [str(embed_dir)], 768, embed_key="clip_l")

    assert load_calls["count"] == 1
    assert torch.equal(first, fake_tensor)
    assert torch.equal(second, fake_tensor)
    sd1_clip._embedding_file_cache.clear()

if __name__ == "__main__":
    try:
        test_expand_directory_list_caching()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
