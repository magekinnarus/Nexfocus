import os
import sys
import torch
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.bgr_engine as bgr_engine

def test_bgr():
    print("Testing BGR Engine...")
    
    # 1. Create a dummy image
    # A simple red square on a black background
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[100:400, 100:400, 0] = 255  # Red box
    
    # 2. Run BGR
    print("Running background removal...")
    try:
        rgba, mask = bgr_engine.remove_background(img, threshold=0.5, jit=True)
        
        print(f"RGBA shape: {rgba.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # 3. Verify shapes
        assert rgba.shape == (512, 512, 4), f"Incorrect RGBA shape: {rgba.shape}"
        assert mask.shape == (512, 512), f"Incorrect mask shape: {mask.shape}"
        
        # 4. Verify mask values (should be binary 0 and 255)
        unique_values = np.unique(mask)
        print(f"Unique mask values: {unique_values}")
        assert all(v in [0, 255] for v in unique_values), f"Mask is not binary: {unique_values}"
        
        # 5. Verify RGBA alpha matches mask
        assert np.array_equal(rgba[:, :, 3], mask), "RGBA alpha channel does not match mask"
        
        print("BGR basic functionality check passed.")
        
        # 6. Test unload
        print("Testing unload_model...")
        bgr_engine.unload_model()
        assert bgr_engine._remover_instance is None, "Model instance was not cleared after unload"
        print("Unload check passed.")
        
        print("\nBGR test successful!")
        return True
        
    except Exception as e:
        print(f"\nBGR test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bgr()
    sys.exit(0 if success else 1)
