import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.flags as flags
from modules.ui_components.metadata_preview import format_metadata_preview


def test_format_metadata_preview_renders_prompt_newlines_without_json_escaping():
    preview = format_metadata_preview(
        {
            'prompt': 'line one\nline two',
            'full_prompt': ['first prompt line\ncontinued prompt line', 'second prompt'],
            'steps': 10,
            'metadata_scheme': 'fooocus_nex',
        },
        flags.MetadataScheme.FOOOCUS_NEX,
    )

    assert '"prompt"' not in preview
    assert '\\n' not in preview
    assert 'prompt:\n    line one\n    line two' in preview
    assert 'steps: 10' in preview
    assert 'full_prompt' not in preview
    assert 'metadata_scheme' not in preview
