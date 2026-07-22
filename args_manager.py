import os
import sys

import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
args_parser.parser.add_argument("--colab", action='store_true',
                                help="Use the Colab config profile (config_colab.txt).")
args_parser.parser.add_argument(
    "--memory-environment-profile",
    type=str,
    default=None,
    help=(
        "Override the resolved memory environment profile. "
        "This changes memory-policy defaults, but does not by itself force Flux Fill streaming posture. "
        "Supported values: auto, colab_pro, colab_free, local_low_vram, local_normal, custom."
    ),
)
args_parser.parser.add_argument(
    "--hardware-total-ram-mb",
    type=float,
    default=None,
    help="Override detected total system RAM (MB) for runtime planning and profile resolution.",
)
args_parser.parser.add_argument(
    "--hardware-total-vram-mb",
    type=float,
    default=None,
    help=(
        "Override detected total GPU VRAM (MB) for runtime planning and profile resolution. "
        "Useful for simulating streaming-class Flux Fill hardware on high-RAM Colab sessions."
    ),
)
args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--skip-model-load", action='store_true',
                                help="Skip loading models at startup (useful for low VRAM)", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

flux_attention_group = args_parser.parser.add_mutually_exclusive_group()
flux_attention_group.add_argument(
    "--flux-attention-backend",
    dest="flux_attention_backend",
    type=str,
    choices=("auto", "sdpa", "xformers", "xformers_only"),
    default=None,
    help=(
        "Select the Flux attention backend policy. "
        "auto keeps the current SDPA default; xformers prefers xformers and falls back to SDPA."
    ),
)
flux_attention_group.add_argument(
    "--flux-attention-auto",
    dest="flux_attention_backend",
    action="store_const",
    const="auto",
    help="Force Flux attention backend policy to auto.",
)
flux_attention_group.add_argument(
    "--flux-attention-sdpa",
    dest="flux_attention_backend",
    action="store_const",
    const="sdpa",
    help="Force Flux attention backend policy to SDPA.",
)
flux_attention_group.add_argument(
    "--flux-attention-xformers",
    dest="flux_attention_backend",
    action="store_const",
    const="xformers",
    help="Prefer xformers for Flux attention, with SDPA fallback.",
)
flux_attention_group.add_argument(
    "--flux-attention-xformers-only",
    dest="flux_attention_backend",
    action="store_const",
    const="xformers_only",
    help="Require xformers for Flux attention and fail if it cannot be used.",
)

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

if "pytest" in sys.modules or any("pytest" in str(arg).lower() for arg in sys.argv[:1]):
    args_parser.args, _ = args_parser.parser.parse_known_args()
else:
    args_parser.args = args_parser.parser.parse_args()

# (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

if getattr(args_parser.args, "port", None) is not None:
    port_val = int(args_parser.args.port)
    if not (1 <= port_val <= 65535):
        raise ValueError("--port must be an integer between 1 and 65535.")

if getattr(args_parser.args, "flux_attention_backend", None):
    os.environ["NEX_FLUX_ATTENTION_BACKEND"] = str(args_parser.args.flux_attention_backend)

args = args_parser.args
