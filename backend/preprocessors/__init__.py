STRUCTURAL_PREPROCESSOR_ASSETS = {
    "Depth": "structural.depth.preprocessor",
}

STRUCTURAL_CONTROLNET_ASSETS = {
    "PyraCanny": "structural.canny.controlnet",
    "CPDS": "structural.cpds.controlnet",
    "Depth": "structural.depth.controlnet",
}

from .runtime import apply_residency_policy, offload_cached_preprocessors, run_structural_preprocessor
