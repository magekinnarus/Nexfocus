import os
import sys
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.blending as blending

def test_blending():
    print("Testing Blending Utility...")
    
    # 1. Test sin_blend_1d
    print("Testing sin_blend_1d...")
    len1d = 100
    w1d = blending.sin_blend_1d(len1d)
    assert w1d.shape == (len1d,)
    assert torch.allclose(w1d[0], torch.tensor(0.0)), f"Start value expected 0.0, got {w1d[0]}"
    assert torch.allclose(w1d[-1], torch.tensor(1.0)), f"End value expected 1.0, got {w1d[-1]}"
    assert torch.allclose(w1d[50], torch.tensor(0.5), atol=1e-2), f"Mid value expected ~0.5, got {w1d[50]}"
    print("sin_blend_1d passed.")

    # 2. Test sin_blend_2d (bell curve)
    print("\nTesting sin_blend_2d...")
    w, h = 64, 128
    w2d = blending.sin_blend_2d(w, h)
    assert w2d.shape == (h, w)
    # Corners should be 0
    assert torch.allclose(w2d[0, 0], torch.tensor(0.0))
    assert torch.allclose(w2d[-1, -1], torch.tensor(0.0))
    # Center should be 1.0
    assert torch.allclose(w2d[h//2, w//2], torch.tensor(1.0), atol=1e-2)
    print("sin_blend_2d bell curve verified.")

    # 3. Test apply_sin2_curve
    print("\nTesting apply_sin2_curve...")
    # Numpy
    arr = np.array([0.0, 0.5, 1.0])
    res_np = blending.apply_sin2_curve(arr)
    assert np.allclose(res_np, [0.0, 0.5, 1.0]), f"Apply curve numpy failed: {res_np}"
    
    # Torch
    ten = torch.tensor([0.0, 0.5, 1.0])
    res_pt = blending.apply_sin2_curve(ten)
    assert torch.allclose(res_pt, torch.tensor([0.0, 0.5, 1.0])), f"Apply curve torch failed: {res_pt}"
    
    print("apply_sin2_curve passed.")

    print("\nBlending utility test successful!")
    return True

if __name__ == "__main__":
    success = test_blending()
    sys.exit(0 if success else 1)
