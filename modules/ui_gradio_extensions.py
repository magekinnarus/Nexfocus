# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import base64
import hashlib
import os
import gradio as gr


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn).replace('\\', '/')

    if os.path.exists(fn):
        return f'file={web_path}?{os.path.getmtime(fn)}'
    return f'file={web_path}'


def read_asset(fn):
    fn = fn.replace('/', os.sep)
    full_path = os.path.normpath(os.path.join(script_path, fn))
    if not os.path.exists(full_path):
        print(f'[UI] Asset not found: {full_path}')
        return ""
    # print(f'[UI] Loading asset: {full_path}')
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_module_assets(folder, extension):
    full_path = os.path.join(script_path, folder.replace('/', os.sep))
    if not os.path.exists(full_path):
        return []

    files = [f for f in os.listdir(full_path)
             if f.endswith(extension) and not f.startswith('.')]
    files.sort()
    return [f'{folder}/{f}' for f in files]


def javascript_html():
    head = creative_renderer_html()
    
    # Load all modules from javascript/modules/ in alphabetical order
    js_files = get_module_assets('javascript/modules', '.js')

    for js_file in js_files:
        content = read_asset(js_file)
        if content:
            head += f'<script type="text/javascript">{content}</script>\n'
    head += '<style>footer { display: none !important; }</style>\n'
    return head


def creative_renderer_html():
    """Load the exact local Konva artifact before all editor modules."""

    renderer_path = os.path.join(script_path, 'javascript', 'vendor', 'konva', '10.6.0', 'konva.min.js')
    expected_hex = 'C03625663B3F4B79C64AECD5671F1A83A37AA2D1E276005B186E42E7D8DBA5A1'
    expected = bytes.fromhex(expected_hex)
    try:
        with open(renderer_path, 'rb') as handle:
            actual = handle.read()
    except OSError:
        actual = b''
    if not actual or hashlib.sha256(actual).digest() != expected:
        return '<script>window.__nexCreativeEditorRendererError = "Pinned Konva 10.6.0 asset is missing or failed its integrity check.";</script>\n'
    integrity = base64.b64encode(expected).decode('ascii')
    return (
        '<script src="/creative_document_api/vendor/konva-10.6.0.js" '
        f'integrity="sha256-{integrity}" crossorigin="anonymous"></script>\n'
    )


def css_html():
    # Base style
    head = f'<style>{read_asset("css/style.css")}</style>\n'
    
    # Module styles
    css_files = get_module_assets('css/modules', '.css')
    for css_file in css_files:
        content = read_asset(css_file)
        if content:
            head += f'<style>{content}</style>\n'
            
    return head


def reload_javascript():
    # Deprecated in Gradio 5.x. Head injection handled via gr.Blocks(head=...)
    pass
