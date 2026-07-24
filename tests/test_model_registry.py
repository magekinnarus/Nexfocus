import os
import sys

sys.argv = [sys.argv[0]]

from modules import model_registry


def test_list_internal_upscale_assets_includes_expected_entries():
    model_registry.clear_asset_index_cache()

    ids = set(model_registry.list_asset_ids(category='upscale', internal_only=True))

    assert 'upscale.4xnomos2_otf_esrgan' in ids
    assert 'upscale.real_hat_gan_srx4' in ids
    assert 'upscale.swin2sr_realworldsr_x4_64_bsrgan_psnr' in ids


def test_resolve_asset_path_uses_destination_roots():
    model_registry.clear_asset_index_cache()

    inpaint_path = model_registry.resolve_asset_path('inpaint.fooocus_patch.v2_6')
    removal_path = model_registry.resolve_asset_path('removals.object.mat.places512')

    assert os.path.normpath(inpaint_path).endswith(os.path.normpath(os.path.join('models', 'inpaint', 'inpaint_v26.fooocus.patch')))
    assert os.path.normpath(removal_path).endswith(os.path.normpath(os.path.join('models', 'removals', 'Places_512_FullData_G.pth')))


def test_flux_fill_assets_are_internal_only_and_resolve_to_expected_roots():
    model_registry.clear_asset_index_cache()

    ids = set(model_registry.list_asset_ids(category='inpaint', engine_family='flux_fill', internal_only=True))

    assert 'inpaint.flux_fill.unet.fp8' in ids
    assert 'inpaint.flux_fill.text_encoder.t5xxl.fp16' in ids
    assert 'inpaint.flux_fill.text_encoder.clip_l' in ids
    assert 'inpaint.flux_fill.ae' in ids

    assert os.path.normpath(model_registry.resolve_asset_path('inpaint.flux_fill.unet.fp8')).endswith(os.path.normpath(os.path.join('models', 'unet', 'flux', 'flux1-Fill-Dev_FP8.safetensors')))
    assert os.path.normpath(model_registry.resolve_asset_path('inpaint.flux_fill.text_encoder.t5xxl.fp16')).endswith(os.path.normpath(os.path.join('models', 'clip', 'flux', 't5xxl_fp16.safetensors')))
    assert os.path.normpath(model_registry.resolve_asset_path('inpaint.flux_fill.ae')).endswith(os.path.normpath(os.path.join('models', 'vae', 'flux', 'ae.safetensors')))
    assert model_registry.get_asset('inpaint.flux_fill.unet.fp8').get('requires') == ['inpaint.flux_fill.ae']
