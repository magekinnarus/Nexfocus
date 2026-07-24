import torch

from backend import conditioning


def test_prompt_conditioning_cache_round_trips_cpu_clones():
    fingerprint = conditioning.build_stage_fingerprint(
        "sdxl_text_conditioning",
        residency_class="cpu",
        prompt="a village",
        negative_prompt="",
        model_identity="model-a",
    )
    payload = {
        "positive": {
            "cond": torch.ones((1, 2, 3), dtype=torch.float32),
            "pooled": torch.ones((1, 3), dtype=torch.float32),
        },
        "negative": {
            "cond": torch.zeros((1, 2, 3), dtype=torch.float32),
            "pooled": torch.zeros((1, 3), dtype=torch.float32),
        },
    }

    conditioning.remember_prompt_conditioning_cache(fingerprint, payload)
    cached = conditioning.load_prompt_conditioning_from_cache(fingerprint)

    assert cached is not None
    assert torch.equal(cached["positive"]["cond"], payload["positive"]["cond"])
    assert torch.equal(cached["positive"]["pooled"], payload["positive"]["pooled"])
    assert torch.equal(cached["negative"]["cond"], payload["negative"]["cond"])
    assert torch.equal(cached["negative"]["pooled"], payload["negative"]["pooled"])

    payload["positive"]["cond"].fill_(99.0)
    assert not torch.equal(cached["positive"]["cond"], payload["positive"]["cond"])
