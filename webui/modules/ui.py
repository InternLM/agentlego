import locale
from pathlib import Path

import gradio as gr
import yaml

with open(Path(__file__).resolve().parent / '../css/main.css', 'r', encoding='utf-8') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../js/main.js', 'r', encoding='utf-8') as f:
    js = f.read()
with open(Path(__file__).resolve().parent / 'i18n.yml', 'r', encoding='utf-8') as f:
    translation = yaml.safe_load(f)

lang = locale.getlocale()[0]
if lang and ('zh_CN' in lang or 'Chinese' in lang):
    lang = 'zh'
else:
    lang = 'en'

refresh_symbol = '🔄'
delete_symbol = '🗑️'
save_symbol = '💾'

theme = gr.themes.Default(
    font=['Noto Sans', 'Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)

agent_elements = {'agent_class': None}


def list_tool_elements():
    elements = [
        'tool_class',
        'tool_name',
        'tool_description',
        'tool_enable',
        'tool_device',
        'tool_args',
    ]
    return elements


def list_interface_input_elements():
    elements = []

    # Chat elements
    elements += [
        'textbox',
        'history',
        'selected_tools',
    ]

    # Tool elements
    elements += list_tool_elements()

    return elements


def gather_interface_values(*args):
    output = {}
    for i, element in enumerate(list_interface_input_elements()):
        output[element] = args[i]

    return output


def apply_interface_values(state):
    elements = list_interface_input_elements()
    if len(state) == 0:
        return [gr.update() for k in elements]  # Dummy, do nothing
    else:
        return [state[k] if k in state else gr.update() for k in elements]


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class, interactive=True):
    """
    Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui
    """
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = gr.Button(refresh_symbol, elem_classes=elem_class, interactive=interactive)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )

    return refresh_button

def create_confirm_cancel(value, **kwargs):
    widget = gr.Button(value, **kwargs)
    hidden = []
    confirm = gr.Button('Confirm', visible=False, **kwargs)
    cancel = gr.Button('Cancel', visible=False, variant='stop', **kwargs)
    hidden.extend([confirm, cancel])

    widget.click(lambda: [gr.update(visible=False)] + [gr.update(visible=True)] * len(hidden), None, [widget, *hidden], show_progress=False)
    cancel.click(lambda: [gr.update(visible=True)] + [gr.update(visible=False)] * len(hidden), None, [widget, *hidden], show_progress=False)

    return widget, confirm, cancel


def get_translator(file_path):
    items = translation.get(Path(file_path).stem, {})
    def trans(key, *args, **kwargs):
        item = items[key].get(lang, 'en')
        return item.format(*args, **kwargs)
    return trans
