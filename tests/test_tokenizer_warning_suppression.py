import os

from backend import clip as backend_clip
from ldm_patched.modules import sd1_clip


class _DummyTokenizer:
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        instance = cls()
        instance.clean_up_tokenization_spaces = None
        return instance

    def __call__(self, _text):
        return {"input_ids": [49406, 49407]}

    def get_vocab(self):
        return {"": 0}


def test_nex_tokenizer_passes_clean_up_tokenization_spaces_flag():
    _DummyTokenizer.from_pretrained_calls.clear()

    backend_clip.NexTokenizer(
        tokenizer_path="dummy-tokenizer",
        tokenizer_class=_DummyTokenizer,
    )

    assert _DummyTokenizer.from_pretrained_calls == [
        (("dummy-tokenizer",), {"clean_up_tokenization_spaces": False})
    ]


def test_sd1_tokenizer_passes_clean_up_tokenization_spaces_flag():
    _DummyTokenizer.from_pretrained_calls.clear()

    sd1_clip.SDTokenizer(
        tokenizer_path=os.path.join("dummy", "tokenizer"),
        tokenizer_class=_DummyTokenizer,
    )

    assert _DummyTokenizer.from_pretrained_calls == [
        ((os.path.join("dummy", "tokenizer"),), {"clean_up_tokenization_spaces": False})
    ]
