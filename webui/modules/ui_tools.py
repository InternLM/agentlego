import inspect
from functools import partial

import gradio as gr

from agentlego.tools.remote import RemoteTool
from . import shared, ui, utils
from .settings import apply_tool_settings_to_state, save_tool_settings
from .tools import delete_tool, load_tool
from .utils import gradio


def create_ui():

    with gr.Tab('Tools', elem_id='tools-tab'):
        with gr.Row():
            shared.gradio['tool_menu'] = gr.Dropdown(choices=utils.get_available_tools(), label='Tools', elem_classes='slim-dropdown')
            ui.create_refresh_button(shared.gradio['tool_menu'], lambda: None, lambda: {'choices': utils.get_available_tools()}, 'refresh-button')
            delete, confirm, cancel = ui.create_confirm_cancel('🗑️', elem_classes='refresh-button')
            shared.gradio['delete_tool'] = delete
            shared.gradio['delete_tool-confirm'] = confirm
            shared.gradio['delete_tool-cancel'] = cancel
            shared.gradio['setup_all_tools'] = gr.Button('Setup All', elem_classes='refresh-button')

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    shared.gradio['tool_class'] = gr.Dropdown(label='Tool class', choices=list(shared.tool_classes), interactive=False, visible=False, elem_classes=['slim-dropdown'])
                    shared.gradio['save_tool'] = gr.Button('Save', visible=False, elem_classes='refresh-button')
                with gr.Group():
                    shared.gradio['tool_name'] = gr.Textbox(label='Name', visible=False)
                    shared.gradio['tool_description'] = gr.Textbox(label='Description', visible=False)
                    shared.gradio['tool_enable'] = gr.Checkbox(label='Enable', value=True, visible=False)
                    shared.gradio['tool_device'] = gr.Dropdown(label='Device', value='cpu', choices=utils.get_available_devices(), visible=False)
                    shared.gradio['tool_args'] = gr.Textbox(label='Initialize arguments', visible=False)

            with gr.Column(scale=1):
                with gr.Row():
                    shared.gradio['remote_server'] = gr.Textbox(label='Import from tool server', elem_classes=['slim-dropdown'])
                    shared.gradio['import_remote'] = gr.Button('Confirm', elem_classes='refresh-button')

                shared.gradio['tool_status'] = gr.Markdown('')


def make_tool_param_visiable(class_name, tool_name):
    if not class_name:
        return [gr.update(visible=False)] * 5

    tool_class = shared.tool_classes[class_name]
    params = inspect.signature(tool_class).parameters
    if 'device' in params:
        device = params['device'].default
        if 'cuda' in str(device) and 'cuda:0' in utils.get_available_devices():
            device = 'cuda:0'
        else:
            device = 'cpu'
    else:
        device = None

    if tool_name == 'New Tool':
        if tool_class is not RemoteTool:
            toolmeta = tool_class.get_default_toolmeta()
            module_name = tool_class.__module__
            if module_name.startswith('_ext_'):
                name = module_name.removeprefix('_ext_') + '.' + toolmeta.name
            else:
                name = toolmeta.name
            description = toolmeta.description
        else:
            name, description = '', ''
        return (
            gr.update(value=name, visible=True),
            gr.update(value=description, visible=True),
            gr.update(value=True, visible=True),
            gr.update(value=device, visible=(device is not None)),
            gr.update(value='', visible=True),
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=(device is not None)),
            gr.update(visible=True),
        )


def setup_tools(*args):
    for name in args:
        tool = shared.toolkits[name]
        if not tool._is_setup:
            yield f'Setup `{name}`...'
            tool.setup()
            tool._is_setup = True
    yield 'Done'


def import_remote_tools(server: str):
    if not server.startswith('http'):
        server = 'http://' + server
    tools = RemoteTool.from_server(server)
    msg = ''
    for tool in tools:
        if tool.name in shared.tool_settings:
            msg += f'- Skip `{tool.name}` since it is already in Tools.\n'
            yield msg
            continue
        save_tool_settings(
            tool_class='RemoteTool',
            name=tool.name,
            desc=tool.toolmeta.description,
            enable=True,
            device='cpu',
            args=f'url="{tool.url}"',
        )
        shared.toolkits[tool.name] = tool
        msg += f'- Imported tool `{tool.name}`\n'
        yield msg


def create_event_handlers():
    def update_widgets(tool_menu):
        if tool_menu is None:
            return [gr.update(visible=False, interactive=False), gr.update(visible=False)]
        elif tool_menu == 'New Tool':
            return [gr.update(visible=True, interactive=True), gr.update(visible=True)]
        else:
            return [gr.update(visible=True, interactive=False), gr.update(visible=True)]

    tool_cfg_widgets = gradio('tool_name', 'tool_description', 'tool_enable', 'tool_device', 'tool_args')

    shared.gradio['tool_menu']\
        .change(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(apply_tool_settings_to_state, gradio('tool_menu', 'interface_state'), gradio('interface_state'))\
        .then(ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False)\
        .then(update_widgets, gradio('tool_menu'), gradio('tool_class', 'save_tool'), show_progress=False)\

    shared.gradio['tool_class'].change(make_tool_param_visiable, gradio('tool_class', 'tool_menu'), tool_cfg_widgets, show_progress=False)

    shared.gradio['setup_all_tools'].click(
        partial(setup_tools, *shared.toolkits), None, gradio('tool_status'))

    def load_tool_wrapper(name):
        yield f'Loading tool `{name}`...'
        try:
            tool = load_tool(name)
        except Exception as e:
            yield f'Failed to load `{name}`'
            raise e
        else:
            if tool is not None:
                yield f'Loaded `{name}`.'
            else:
                yield f'Skipped disabled tool `{name}`.'

    shared.gradio['save_tool']\
        .click(save_tool_settings, gradio('tool_class') + tool_cfg_widgets + gradio('tool_menu'), gradio('tool_status'), show_progress=False)\
        .success(load_tool_wrapper, gradio('tool_name'), gradio('tool_status'))\
        .success(lambda name: gr.update(value=name, choices=utils.get_available_tools()), gradio('tool_name'), gradio('tool_menu'))\
        .then(lambda: gr.update(choices=list(shared.toolkits)), None, gradio('selected_tools'))

    shared.gradio['import_remote']\
        .click(import_remote_tools, gradio('remote_server'), gradio('tool_status'))\
        .then(lambda: gr.update(choices=utils.get_available_tools()), None, gradio('tool_menu'))\
        .then(lambda: gr.update(choices=list(shared.toolkits)), None, gradio('selected_tools'))

    delete_tool_widgets = ('delete_tool', 'delete_tool-confirm', 'delete_tool-cancel')
    shared.gradio['delete_tool-confirm']\
        .click(delete_tool, gradio('tool_menu'), gradio('tool_status'))\
        .success(lambda: gr.update(value=None, choices=utils.get_available_tools()), None, gradio('tool_menu'))\
        .then(lambda: gr.update(choices=list(shared.toolkits)), None, gradio('selected_tools'))\
        .then(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, gradio(delete_tool_widgets))
