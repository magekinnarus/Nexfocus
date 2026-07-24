from __future__ import annotations

import torch

from backend import cond_utils
from ldm_patched.modules import conds as ldm_conds


class _DummyModel:
    def get_dtype(self):
        return torch.float16

    def extra_conds(self, **kwargs):
        result = {}
        if "cross_attn" in kwargs:
            result["c_crossattn"] = ldm_conds.CONDCrossAttn(kwargs["cross_attn"])
        if "pooled_output" in kwargs:
            result["y"] = ldm_conds.CONDRegular(kwargs["pooled_output"])
        return result


class _NoExtraCondsModel:
    def get_dtype(self):
        return torch.float16


def test_process_conds_stages_prepared_conditioning_once():
    device = torch.device("cpu")
    noise = torch.zeros((1, 4, 8, 8), device=device, dtype=torch.float16)
    cross_attn = torch.randn((1, 77, 2048), dtype=torch.float16)
    pooled = torch.randn((1, 1280), dtype=torch.float16)

    conds = {
        "positive": [
            {
                "cross_attn": cross_attn,
                "pooled_output": pooled,
                "model_conds": {},
            }
        ],
        "negative": [],
    }

    processed = cond_utils.process_conds(_DummyModel(), noise, conds, device)
    positive = processed["positive"][0]

    assert "cross_attn" not in positive
    assert "c_crossattn" in positive["model_conds"]
    assert positive["model_conds"]["c_crossattn"].cond.device == device
    assert positive["model_conds"]["y"].cond.device == device


def test_process_conds_stages_raw_conditioning_when_model_conds_not_present():
    device = torch.device("cpu")
    noise = torch.zeros((1, 4, 8, 8), device=device, dtype=torch.float16)
    cross_attn = torch.randn((1, 77, 2048), dtype=torch.float32)

    conds = {
        "positive": [
            {
                "cross_attn": cross_attn,
                "model_conds": {},
            }
        ],
        "negative": [],
    }

    processed = cond_utils.process_conds(_NoExtraCondsModel(), noise, conds, device)
    positive = processed["positive"][0]

    assert positive["cross_attn"].device == device
    assert positive["cross_attn"].dtype == torch.float16
