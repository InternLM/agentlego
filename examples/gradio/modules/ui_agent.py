from functools import partial

import gradio as gr

from . import shared, ui, utils
from .agents import agent_func_map, delete_agent, load_llm, unload_agent
from .logging import logger
from .settings import (apply_agent_settings, make_agent_params_visible,
                       save_agent_settings)
from .utils import gradio


def create_ui():

    with gr.Tab('Agent', elem_id='agent-tab'):
        with gr.Row():
            shared.gradio['agent_menu'] = gr.Dropdown(choices=utils.get_available_agents(), label='Agent', elem_classes='slim-dropdown')
            ui.create_refresh_button(shared.gradio['agent_menu'], lambda: None, lambda: {'choices': utils.get_available_agents()}, 'refresh-button')
            delete, confirm, cancel = ui.create_confirm_cancel('🗑️', elem_classes='refresh-button')
            shared.gradio['delete_agent'] = delete
            shared.gradio['delete_agent-confirm'] = confirm
            shared.gradio['delete_agent-cancel'] = cancel
            shared.gradio['load_agent'] = gr.Button('Load', elem_classes='refresh-button')

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    shared.gradio['agent_class'] = gr.Dropdown(label='Agent class', choices=agent_func_map.keys(), value=None, elem_classes=['slim-dropdown'], scale=3, interactive=False)
                    shared.gradio['save_agent'] = gr.Button('Save', visible=False, elem_classes='refresh-button')
                    shared.gradio['save_agent_new'] = gr.Button('Save to', visible=False, elem_classes='refresh-button')
                    shared.gradio['new_agent_name'] = gr.Textbox(label='Agent name', visible=False, scale=1, elem_classes=['slim-dropdown'])
                with gr.Group():
                    # Agent initialization arguments
                    for agent_class, callbacks in agent_func_map.items():
                        for name, widget in callbacks.cfg_widget().items():
                            widget.visible = False
                            name = f'{agent_class}#{name}'
                            shared.gradio[name] = widget
                            ui.agent_elements.append(name)

            with gr.Column(scale=1):
                shared.gradio['agent_status'] = gr.Markdown('No agent is loaded' if shared.agent_name == 'None' else 'Ready')


def create_event_handlers():
    def update_widgets(agent_menu):
        if agent_menu == 'New Agent':
            return [gr.update(visible=True, interactive=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
        elif agent_menu is None:
            return [gr.update(visible=False, interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        else:
            return [gr.update(visible=True, interactive=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

    def update_current_agent():
        if shared.agent_name is None:
            return '<div class="current-agent-warn">No agent is loaded, please select in the Agent tab.</div>'
        else:
            return f'<div class="current-agent">Current agent: {shared.agent_name}</div>'

    shared.gradio['agent_menu']\
        .change(apply_agent_settings, gradio('agent_menu'), gradio(ui.agent_elements))\
        .then(update_widgets, gradio('agent_menu'), gradio('agent_class', 'save_agent', 'save_agent_new', 'new_agent_name'), show_progress=False)\
        .then(load_agent, gradio('agent_menu'), gradio('agent_status'), show_progress=False)\
        .success(update_current_agent, None, gradio('current-agent'), show_progress=False)\

    shared.gradio['agent_class'].change(make_agent_params_visible, gradio('agent_class'), gradio(ui.agent_elements), show_progress=False)

    shared.gradio['load_agent']\
        .click(partial(load_agent, autoload=True), gradio('agent_menu'), gradio('agent_status'), show_progress=False)\
        .success(update_current_agent, None, gradio('current-agent'), show_progress=False)\

    shared.gradio['save_agent']\
        .click(save_agent_settings, gradio('agent_menu', *ui.agent_elements), gradio('agent_status'), show_progress=False)

    shared.gradio['save_agent_new']\
        .click(save_agent_settings, gradio('new_agent_name', *ui.agent_elements), gradio('agent_status'), show_progress=False)\
        .success(lambda name: gr.update(value=name, choices=utils.get_available_agents()), gradio('new_agent_name'), gradio('agent_menu'))\
        .then(lambda: gr.update(value=''), None, gradio('new_agent_name'))\

    delete_agent_widgets = ('delete_agent', 'delete_agent-confirm', 'delete_agent-cancel')
    shared.gradio['delete_agent-confirm']\
        .click(delete_agent, gradio('agent_menu'), gradio('agent_status'))\
        .success(lambda: gr.update(value=None, choices=utils.get_available_agents()), None, gradio('agent_menu'))\
        .then(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, gradio(delete_agent_widgets))


def load_agent(selected_agent, autoload=False):
    if selected_agent is None:
        yield 'No agent selected'
    elif selected_agent == 'New Agent':
        yield 'Please save the new agent before load it.'
    elif not autoload:
        if shared.agent_name is None:
            yield f"Click on \"Load\" to load `{selected_agent}`."
        elif shared.agent_name != selected_agent:
            yield f"The current agent is `{shared.agent_name}`.\n\nClick on \"Load\" to load `{selected_agent}`."
        else:
            yield f'The agent `{shared.agent_name}` is loaded.'
        return
    else:
        try:
            yield f'Loading `{selected_agent}`...'
            unload_agent()
            shared.agent_name = selected_agent
            shared.llm = load_llm(selected_agent)

            if shared.llm is not None:
                yield f'Successfully loaded `{selected_agent}`.'
            else:
                yield f'Failed to load `{selected_agent}`.'
        except:
            logger.exception('Traceback')
            raise gr.Error(f'Failed to load the agent `{selected_agent}`.')
