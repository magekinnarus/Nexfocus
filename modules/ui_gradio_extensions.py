# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


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
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    
    # Load all modules from javascript/modules/ in alphabetical order
    js_files = get_module_assets('javascript/modules', '.js')

    for js_file in js_files:
        content = read_asset(js_file)
        if content:
            head += f'<script type="text/javascript">{content}</script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'

    head += '<style>footer { display: none !important; }</style>\n'
    return head


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
