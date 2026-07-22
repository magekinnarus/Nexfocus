import math
import warnings

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import ldm_patched.ldm.modules.attention as attention
import backend.resources as runtime_resources

import backend.ip_adapter as contextual_ip_adapter


eva_clip_models = {}
face_parsers = {}


def _offload_module(module):
    if module is None:
        return
    try:
        module.to('cpu')
    except Exception:
        pass


def apply_contextual_residency(mode='offload'):
    global eva_clip_models, face_parsers

    for module in eva_clip_models.values():
        _offload_module(module)
    for parser in face_parsers.values():
        _offload_module(parser)

    actions = {
        'mode': mode,
        'eva_clip_models': len(eva_clip_models),
        'face_parsers': len(face_parsers),
    }

    if mode == 'destroy':
        eva_clip_models = {}
        face_parsers = {}

    return actions


def release_preprocess_support(*, reclaim_device_memory=True):
    """Release all contextual support after the reusable CPU payload exists."""
    actions = apply_contextual_residency('destroy')
    actions['contextual'] = contextual_ip_adapter.release_contextual_preprocess_support(
        reclaim_device_memory=reclaim_device_memory,
    )
    return actions


def has_preprocess_support():
    return bool(
        eva_clip_models
        or face_parsers
        or contextual_ip_adapter.has_contextual_preprocess_support()
    )


def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255.0, 0, 1)
    return tensor[..., [2, 1, 0]]


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return x.repeat(1, 3, 1, 1)


def load_eva_clip(eva_clip_path):
    if not isinstance(eva_clip_path, str):
        raise RuntimeError('PuLID requires an EVA-CLIP model path.')

    if eva_clip_path in eva_clip_models:
        return eva_clip_models[eva_clip_path]

    from backend.eva_clip.factory import create_model_and_transforms

    model, _, _ = create_model_and_transforms(
        'EVA02-CLIP-L-14-336',
        pretrained=eva_clip_path,
        force_custom_clip=True,
        device='cpu',
    )
    visual = model.visual
    eva_clip_models[eva_clip_path] = visual
    return visual


def load_face_parser(device=None):
    import modules.config as config
    from extras.facexlib.parsing import init_parsing_model

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache_key = str(device)
    if cache_key in face_parsers:
        return face_parsers[cache_key]

    parser = init_parsing_model(
        model_name='bisenet',
        device=device,
        model_rootpath=config.path_insightface,
    )
    face_parsers[cache_key] = parser
    return parser


def _detect_faces(face_app, bgr_image):
    for size in range(640, 256, -64):
        face_app.det_model.input_size = (size, size)
        faces = face_app.get(bgr_image)
        if faces:
            return sorted(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
                reverse=True,
            )
    return []


@torch.no_grad()
@torch.inference_mode()
def preprocess(img, model_path, eva_clip_path, insightface_model_names=None):
    if has_preprocess_support():
        release_preprocess_support(reclaim_device_memory=True)
    try:
        return _preprocess(img, model_path, eva_clip_path, insightface_model_names)
    finally:
        if has_preprocess_support():
            release_preprocess_support(reclaim_device_memory=True)


def _preprocess(img, model_path, eva_clip_path, insightface_model_names=None):
    try:
        from backend.sdxl_unified_runtime import _PREPROCESSOR_METRICS
    except ImportError:
        _PREPROCESSOR_METRICS = None

    cache_key = contextual_ip_adapter._build_contextual_payload_cache_key(
        img,
        cn_type="pulid",
        model_path=model_path,
        clip_vision_path=eva_clip_path,
        ip_negative_path=None,
        insightface_model_names=insightface_model_names,
    )

    cached_val = contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE.get(cache_key)
    if cached_val is not None:
        contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE.move_to_end(cache_key)
        if _PREPROCESSOR_METRICS is not None:
            _PREPROCESSOR_METRICS["contextual_hits"] += 1.0
        return contextual_ip_adapter._clone_contextual_payload(cached_val)

    if _PREPROCESSOR_METRICS is not None:
        _PREPROCESSOR_METRICS["contextual_misses"] += 1.0

    from insightface.utils import face_align

    entry = contextual_ip_adapter.load_contextual_model(model_path)
    if entry is None:
        raise RuntimeError('PuLID model is missing its contextual checkpoint path.')

    adapter_model = entry['model']
    load_device = adapter_model.load_device
    dtype = adapter_model.dtype

    try:
        eva_clip = load_eva_clip(eva_clip_path)
        face_parser = load_face_parser(device=load_device)

        model_names = insightface_model_names or ['antelopev2']
        bgr_image = np.ascontiguousarray(img[:, :, ::-1])
        faces = []
        for model_name in model_names:
            face_app = contextual_ip_adapter.load_insightface(model_name=model_name)
            faces = _detect_faces(face_app, bgr_image)
            if faces:
                break

        if not faces:
            raise RuntimeError('PuLID preprocessing could not detect a face in the reference image.')

        eva_clip = eva_clip.to(load_device, dtype=dtype)
        runtime_resources.load_model_gpu(entry['image_proj_model'])

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond_embeddings = []
        uncond_embeddings = []

        image_size = eva_clip.image_size if isinstance(eva_clip.image_size, int) else eva_clip.image_size[0]
        interpolation = T.InterpolationMode.BICUBIC if 'cuda' in load_device.type else T.InterpolationMode.BILINEAR

        for face in faces:
            iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(load_device, dtype=dtype)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="`estimate` is deprecated since version 0.26",
                    category=FutureWarning,
                )
                aligned = face_align.norm_crop(bgr_image, landmark=face.kps, image_size=512)
            face_tensor = image_to_tensor(aligned).unsqueeze(0).permute(0, 3, 1, 2).to(load_device)

            parsing_input = TF.normalize(face_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            parsing_out = face_parser(parsing_input)[0].argmax(dim=1, keepdim=True)
            bg = sum((parsing_out == label) for label in bg_label).bool()
            white_image = torch.ones_like(face_tensor)
            face_features_image = torch.where(bg, white_image, to_gray(face_tensor))
            face_features_image = TF.resize(face_features_image, [image_size, image_size], interpolation=interpolation)
            face_features_image = TF.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std).to(load_device, dtype=dtype)

            id_cond_vit, id_vit_hidden = eva_clip(
                face_features_image,
                return_all_features=False,
                return_hidden=True,
                shuffle=False,
            )
            id_cond_vit = id_cond_vit.to(load_device, dtype=dtype)
            id_vit_hidden = [hidden.to(load_device, dtype=dtype) for hidden in id_vit_hidden]
            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            id_uncond = torch.zeros_like(id_cond)
            id_hidden_uncond = [torch.zeros_like(hidden) for hidden in id_vit_hidden]

            cond_embeddings.append(entry['image_proj_model'].model(id_cond, id_vit_hidden))
            uncond_embeddings.append(entry['image_proj_model'].model(id_uncond, id_hidden_uncond))

        cond = torch.mean(torch.cat(cond_embeddings, dim=0), dim=0, keepdim=True).to(load_device, dtype=dtype)
        uncond = torch.mean(torch.cat(uncond_embeddings, dim=0), dim=0, keepdim=True).to(load_device, dtype=dtype)

        zero_tensor = torch.zeros((cond.size(0), 8, cond.size(-1)), dtype=dtype, device=load_device)
        cond = torch.cat([cond, zero_tensor], dim=1)
        uncond = torch.cat([uncond, zero_tensor], dim=1)

        runtime_resources.load_model_gpu(entry['ip_layers'])
        kv_modules = contextual_ip_adapter._sorted_kv_modules(entry['ip_layers'].model)
        ip_conds = [module(cond).cpu() for module in kv_modules]
        ip_unconds = [module(uncond).cpu() for module in kv_modules]
        
        res = (ip_conds, ip_unconds)

        contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE[cache_key] = contextual_ip_adapter._clone_contextual_payload(res)
        contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE.move_to_end(cache_key)
        while len(contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE) > contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE_LIMIT:
            contextual_ip_adapter._CONTEXTUAL_PAYLOAD_CACHE.popitem(last=False)

        return res

    finally:
        for module in eva_clip_models.values():
            _offload_module(module)
        for parser in face_parsers.values():
            _offload_module(parser)
        contextual_ip_adapter._offload_contextual_entry(entry)


@torch.no_grad()
@torch.inference_mode()
def patch_model(model, tasks):
    new_model = model.clone()

    def normalize_task(task):
        if len(task) >= 4:
            return task[0], float(task[1]), float(task[2]), float(task[3])
        if len(task) == 3:
            return task[0], float(task[1]), float(task[2]), 0.0
        raise ValueError(f"Unexpected contextual task shape: {task!r}")

    def make_attn_patcher(ip_index):
        def patcher(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            current_step = float(model.model.diffusion_model.current_step.detach().cpu().numpy()[0])
            cond_or_uncond = extra_options['cond_or_uncond']

            q = n
            out = attention.optimized_attention(q, context_attn2, value_attn2, heads=extra_options['n_heads'], mask=None)

            for task in tasks:
                (cs, ucs), cn_stop, cn_weight, cn_start = normalize_task(task)
                if current_step < cn_start or current_step >= cn_stop:
                    continue

                ip_k_c = cs[ip_index * 2].to(q)
                ip_v_c = cs[ip_index * 2 + 1].to(q)
                ip_k_uc = ucs[ip_index * 2].to(q)
                ip_v_uc = ucs[ip_index * 2 + 1].to(q)

                ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0)
                out_ip = attention.optimized_attention(q, ip_k, ip_v, heads=extra_options['n_heads'], mask=None)

                out_fp = out.to(dtype=torch.float32)
                out_ip_fp = out_ip.to(dtype=torch.float32)
                attn_map = q.to(dtype=torch.float32) @ ip_k.transpose(-2, -1).to(dtype=torch.float32)
                attn_mean = attn_map.softmax(dim=-1).mean(dim=1, keepdim=True)
                attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
                projection = (
                    torch.sum((out_fp * out_ip_fp), dim=-2, keepdim=True)
                    / torch.sum((out_fp * out_fp), dim=-2, keepdim=True).clamp(min=1e-6)
                    * out_fp
                )
                orthogonal = out_ip_fp + (attn_mean - 1) * projection
                out = out_fp + cn_weight * orthogonal

            return out.to(dtype=org_dtype)

        return patcher

    def set_model_patch_replace(model_clone, number, key):
        transformer_options = model_clone.model_options['transformer_options']
        if 'patches_replace' not in transformer_options:
            transformer_options['patches_replace'] = {}
        if 'attn2' not in transformer_options['patches_replace']:
            transformer_options['patches_replace']['attn2'] = {}
        if key not in transformer_options['patches_replace']['attn2']:
            transformer_options['patches_replace']['attn2'][key] = make_attn_patcher(number)

    number = 0
    for block_id in [4, 5, 7, 8]:
        block_indices = range(2) if block_id in [4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ('input', block_id, index))
            number += 1

    for block_id in range(6):
        block_indices = range(2) if block_id in [3, 4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ('output', block_id, index))
            number += 1

    for index in range(10):
        set_model_patch_replace(new_model, number, ('middle', 1, index))
        number += 1

    return new_model
