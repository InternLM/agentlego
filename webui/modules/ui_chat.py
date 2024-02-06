from datetime import datetime
from functools import partial

import gradio as gr

from . import chat, shared, ui
from .html_generator import chat_html_wrapper
from .text_generation import stop_everything_event
from .utils import gradio

i18n = ui.get_translator(__file__)

def create_ui():
    shared.gradio['Chat input'] = gr.State()
    shared.gradio['uploaded-files'] = gr.State([])
    shared.gradio['history'] = gr.State({'internal': [], 'visible': []})

    with gr.Tab('Chat', elem_id='chat-tab'):
        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                # Display history
                shared.gradio['current-agent'] = gr.HTML(value=f'<div class="current-agent-warn">{i18n("agent_warn")}</div>')
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}))

                # Chat input area
                with gr.Row(elem_id='chat-input-row'):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.Textbox(label='', placeholder=i18n('placeholder'), elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='upload-container'):
                        shared.gradio['upload-file'] = gr.UploadButton('📁', elem_id='upload', file_types=['image', 'audio'])

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button(i18n('stop'), elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button(i18n('generate'), elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                shared.gradio['Regenerate'] = gr.Button(i18n('regenerate'), elem_id='Regenerate')
                shared.gradio['Modify last'] = gr.Button(i18n('modify_last'), elem_id='Modify-last')
                shared.gradio['Start new chat'] = gr.Button(i18n('start_new_chat'))

        with gr.Group():
            gr.HTML(f'<div style="text-align:center;font-weight:bold;padding:6px">{i18n("select_tools")}</div>')
            with gr.Row():
                shared.gradio['select_all_tools'] = gr.Button('All', size='sm')
                shared.gradio['select_no_tools'] = gr.Button('None', size='sm')
            shared.gradio['selected_tools'] = gr.CheckboxGroup(show_label=False, choices=shared.toolkits.keys(), value=list(shared.toolkits.keys()))

        with gr.Row(elem_id='past-chats-row'):
            shared.gradio['unique_id'] = gr.Dropdown(label=i18n('past_chat'), elem_classes=['slim-dropdown'], choices=chat.find_all_histories(), value=None, allow_custom_value=True)
            shared.gradio['save_chat'] = gr.Button('Save', elem_classes='refresh-button')
            delete, confirm, cancel = ui.create_confirm_cancel('🗑️', elem_classes='refresh-button')
            shared.gradio['delete_chat'] = delete
            shared.gradio['delete_chat-confirm'] = confirm
            shared.gradio['delete_chat-cancel'] = cancel

def create_event_handlers():

    inputs = ('Chat input', 'interface_state', 'uploaded-files')

    shared.gradio['Generate']\
        .click(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False)\
        .then(chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False)\
        .then(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(lambda: [], None, gradio('uploaded-files'), show_progress=False)

    shared.gradio['textbox']\
        .submit(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False)\
        .then(chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False)\
        .then(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(lambda: [], None, gradio('uploaded-files'), show_progress=False)

    shared.gradio['upload-file']\
        .upload(chat.upload_file, gradio('upload-file', 'uploaded-files', 'history'), gradio('uploaded-files', 'display'), show_progress=False)\
        .then(lambda: gr.update(value=None), None, gradio('upload-file'), show_progress=False)

    shared.gradio['Stop']\
        .click(stop_everything_event, None, None, queue=False)\
        .then(chat.redraw_html, gradio('history'), gradio('display'))

    shared.gradio['select_all_tools'].click(lambda: list(shared.toolkits), None, gradio('selected_tools'), show_progress=False)
    shared.gradio['select_no_tools'].click(lambda: [], None, gradio('selected_tools'), show_progress=False)

    shared.gradio['unique_id']\
        .select(chat.load_history, gradio('unique_id'), gradio('history'))\
        .then(chat.redraw_html, gradio('history'), gradio('display'))

    shared.gradio['Start new chat']\
        .click(chat.start_new_chat, None, gradio('history'))\
        .then(chat.redraw_html, gradio('history'), gradio('display'))\
        .then(lambda: gr.update(value=None), None, gradio('unique_id'))

    shared.gradio['Regenerate']\
        .click(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False)\
        .then(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))

    shared.gradio['Modify last']\
        .click(ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state'))\
        .then(chat.modify_last_message, gradio('history'), gradio('textbox', 'history'), show_progress=False)\
        .then(chat.redraw_html, gradio('history'), gradio('display'))

    shared.gradio['save_chat']\
        .click(lambda unique_id: unique_id or datetime.now().strftime('%Y%m%d-%H-%M-%S'), gradio('unique_id'), gradio('unique_id'))\
        .then(chat.save_history, gradio('history', 'unique_id'), None)\
        .then(lambda: gr.update(choices=chat.find_all_histories()), None, gradio('unique_id'), show_progress=False)

    delete_history_widgets = ('delete_chat', 'delete_chat-confirm', 'delete_chat-cancel')
    shared.gradio['delete_chat-confirm']\
        .click(lambda: {'internal': [], 'visible': []}, None, gradio('history'))\
        .then(chat.redraw_html, gradio('history'), gradio('display'))\
        .then(chat.delete_history, gradio('unique_id'), None)\
        .then(lambda: gr.update(value=None, choices=chat.find_all_histories()), None, gradio('unique_id'), show_progress=False)\
        .then(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, gradio(delete_history_widgets))
