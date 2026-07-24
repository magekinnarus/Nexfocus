from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_module(name):
    return (REPO_ROOT / "javascript" / "modules" / name).read_text(encoding="utf-8")


def test_image_slot_uses_click_only_file_browser_activation():
    source = _read_module("40_nex_image_slot.js")

    assert "this.dropZone.addEventListener('click'" in source
    assert "this.dropZone.addEventListener('keydown'" not in source
    assert 'class="nex-slot__drop" tabindex=' not in source


def test_mask_hotkeys_use_the_director_approved_bindings():
    source = _read_module("10_inpaint_mask.js")

    assert 'if (key === "r")' in source
    assert 'if (key === "e")' in source
    assert 'if (key === "a")' in source
    assert 'if (key === "q" || key === "w")' in source
    assert 'if (key === "f")' in source
    assert 'if (key === "b")' not in source
    assert 'if (key === "c")' not in source


def test_escape_handlers_respect_overlay_priority():
    staging_source = _read_module("20_staging_viewer.js")
    compare_source = _read_module("45_nex_image_compare.js")

    assert "markerPicker.style.display !== 'none'" in staging_source
    assert "event.preventDefault();" in staging_source
    assert "if (event.defaultPrevented)" in compare_source


def test_dropped_face_grid_frontend_is_not_auto_injected_or_styled():
    face_grid_module = (
        REPO_ROOT / "javascript" / "modules" / "50_nex_face_grid_composer.js"
    )
    image_slot_css = (
        REPO_ROOT / "css" / "modules" / "12_nex_image_slot.css"
    ).read_text(encoding="utf-8")

    assert not face_grid_module.exists()
    assert "face-grid" not in image_slot_css
    assert "nex-face-grid" not in image_slot_css


def test_replace_bb_image_activates_step2_bb_mode():
    source = _read_module("10_inpaint_mask.js")

    assert 'setActiveMode("bb");' in source
    assert 'setStatus("Refreshing BB image...");' in source
    assert 'setStatus("Step 2 Inpaint BB mask ready.", mode);' in source


def test_mask_hotkeys_yield_to_modal_overlays():
    mask_source = _read_module("10_inpaint_mask.js")
    compare_source = _read_module("45_nex_image_compare.js")

    assert "isModalOverlayActive" in mask_source
    assert 'document.getElementById("nex-compare-overlay")' in mask_source
    assert 'key === "r"' in compare_source
    assert "resetCamera();" in compare_source


