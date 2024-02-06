import time
from threading import Lock

import gradio as gr
from modules import chat, shared, ui, ui_agent, ui_chat, ui_tools, utils
from modules.agents import load_llm
from modules.logging import logger
from modules.tools import load_tool
from modules.utils import gradio


def create_interface():

    title = 'AgentLego Web UI'

    # Password authentication
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding='utf8') as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    # Interface state elements
    shared.input_elements = ui.list_interface_input_elements()

    with gr.Blocks(
            css=ui.css,
            analytics_enabled=False,
            title=title,
            theme=ui.theme,
    ) as shared.gradio['interface']:

        # Interface state
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        ui_chat.create_ui()  # Chat Tab
        ui_agent.create_ui()  # Agent Tab
        ui_tools.create_ui()  # Tools Tab

        ui_chat.create_event_handlers()
        ui_agent.create_event_handlers()
        ui_tools.create_event_handlers()

        shared.gradio['interface'].load(lambda: None, None, None, js="() => document.getElementsByTagName('body')[0].classList.add('dark')")
        shared.gradio['interface'].load(lambda: None, None, None, js=f'() => {{{ui.js}}}')
        shared.gradio['interface'].load(lambda: shared.agent_name, None, gradio('agent_menu'), show_progress=False)
        shared.gradio['interface'].load(lambda: gr.update(choices=chat.find_all_histories()), None, gradio('unique_id'), show_progress=False)
        shared.gradio['interface'].load(chat.redraw_html, gradio('history'), gradio('display'))

    # Launch the interface
    shared.gradio['interface'].queue()
    shared.gradio['interface'].launch(
        prevent_thread_lock=True,
        share=shared.args.share,
        server_name=None if not shared.args.listen else (shared.args.listen_host or '0.0.0.0'),
        server_port=shared.args.listen_port,
        inbrowser=shared.args.auto_launch,
        auth=auth or None,
        ssl_verify=False if (shared.args.ssl_keyfile or shared.args.ssl_certfile) else True,
        ssl_keyfile=shared.args.ssl_keyfile,
        ssl_certfile=shared.args.ssl_certfile,
        max_threads=64,
        allowed_paths=['.'],
    )


if __name__ == '__main__':

    # Initialize agent
    available_agents = utils.get_available_agents()
    if shared.args.agent is not None:
        assert shared.args.agent in available_agents
        shared.agent_name = shared.args.agent
    if shared.agent_name is not None:
        # Load the agent
        shared.llm = load_llm(shared.agent_name)

    # Initialize tools
    for name in shared.tool_settings:
        try:
            logger.info(f'Loading tool `{name}`')
            load_tool(name)
        except Exception:
            logger.exception('Traceback')
            logger.warning(f'Failed to load tool `{name}`, auto disabled.')

    shared.generation_lock = Lock()

    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
