import os
import sys

sys.argv = [sys.argv[0]]
import torch
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.objr_engine as objr_engine

def test_get_segments_keep_full_tile_size_for_edge_cases():
    for length in [833, 896, 1000, 1792, 2400]:
        segments = objr_engine.get_segments(length, tile_size=512, overlap=64)
        assert segments, f"No segments returned for length {length}"

        for start, end, pad_l, _pad_r in segments:
            tile_start = start - pad_l
            tile_end = min(length, tile_start + 512)
            tile_len = tile_end - tile_start
            assert tile_len == 512, (
                f"Length {length} produced undersized tile {tile_len} "
                f"from segment {(start, end, pad_l)}"
            )

def test_objr():
    print("Testing OBJR Engine (MAT)...")
    
    # 1. Test Model Loading
    print("Testing model loading...")
    try:
        model = objr_engine.load_model()
        print("Model loaded and state_dict remapped successfully.")
    except Exception as e:
        print(f"Model loading FAILED: {e}")
        return False

    # 2. Test Small Image (512x512)
    print("\nTesting small image (512x512)...")
    img_512 = np.zeros((512, 512, 3), dtype=np.uint8)
    img_512[200:300, 200:300, 0] = 255 # Red box
    mask_512 = np.zeros((512, 512), dtype=np.uint8)
    mask_512[220:280, 220:280] = 255 # Mask inside red box
    
    try:
        res_512 = objr_engine.remove_object(img_512, mask_512)
        print(f"Small image result shape: {res_512.shape}")
        assert res_512.shape == (512, 512, 3)
        
        # Check that non-masked area is unchanged (approx)
        diff = np.abs(img_512.astype(float) - res_512.astype(float))
        # Masked area should be changed, unmasked should be same
        mask_3d = np.stack([mask_512]*3, axis=-1)
        unmasked_diff = diff * (1 - mask_3d/255.0)
        max_unmasked_diff = np.max(unmasked_diff)
        print(f"Max difference in unmasked area: {max_unmasked_diff}")
        assert max_unmasked_diff < 1.0, f"Unmasked area was modified: {max_unmasked_diff}"
        
        print("Small image test passed.")
    except Exception as e:
        print(f"Small image test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Test Large Image (Tiled, 768x768)
    print("\nTesting large image (768x768 tiling)...")
    img_768 = np.zeros((768, 768, 3), dtype=np.uint8)
    img_768[300:400, 300:400, 1] = 255 # Green box
    mask_768 = np.zeros((768, 768), dtype=np.uint8)
    mask_768[320:380, 320:380] = 255 # Mask inside green box
    
    try:
        res_768 = objr_engine.remove_object(img_768, mask_768)
        print(f"Large image result shape: {res_768.shape}")
        assert res_768.shape == (768, 768, 3)
        
        # Check unmasked area
        diff = np.abs(img_768.astype(float) - res_768.astype(float))
        mask_3d = np.stack([mask_768]*3, axis=-1)
        unmasked_diff = diff * (1 - mask_3d/255.0)
        max_unmasked_diff = np.max(unmasked_diff)
        print(f"Max difference in unmasked area (large): {max_unmasked_diff}")
        assert max_unmasked_diff < 1.0, f"Unmasked area in large image was modified: {max_unmasked_diff}"
        
        print("Large image test passed.")
    except Exception as e:
        print(f"Large image test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test Unload
    print("\nTesting unload_model...")
    objr_engine.unload_model()
    assert objr_engine._model_instance is None, "Model instance not cleared"
    print("Unload test passed.")

    print("\nOBJR (MAT) test successful!")
    return True

if __name__ == "__main__":
    success = test_objr()
    sys.exit(0 if success else 1)
