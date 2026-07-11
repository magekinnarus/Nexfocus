from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLAssemblyResult,
    SDXLRuntimeIdentity,
    ResolvedFileIdentity,
    SDXLLoraSpec,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SDXLAssemblyEligibilityError,
    SDXLAssemblyValidationError,
    ColorExtractionSpec,
)
from backend.sdxl_assembly.color_extraction_worker import ColorExtractionWorker
from backend.sdxl_assembly.wavelet_color import wavelet_reconstruction, wavelet_decomposition
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.assembly import SDXLAssembly
from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
from backend.sdxl_assembly.runtime_state import (
    clear_all_caches,
    release_model_prompt_caches,
    release_prompt_conditioning_caches,
    release_spatial_vae_caches,
    LifecycleDomain,
    release_domain,
)
from backend.sdxl_assembly.lifecycle_coordinator import (
    LifecycleChange,
    LifecycleReleasePlan,
    LifecycleReleaseResult,
    plan_release_for_changes,
    release_for_changes,
)
