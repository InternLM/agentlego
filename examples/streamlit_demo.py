# flake8: noqa
import argparse
import base64
import copy
import hashlib
import re
import uuid
from datetime import datetime
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Tuple

import streamlit as st
from lagent.actions import ActionExecutor
from lagent.agents.react import ReAct, ReActProtocol
from lagent.schema import ActionReturn, AgentReturn
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFileRec

from agentlego import list_tools, load_tool


def load_image(path):
    return UploadedFileRec(
        0, Path(path).name, type='image', data=open(path, 'rb').read())


logger = getLogger('AgentLego')
IMG_REGEX = r'(?P<path>([/\.][\w_-]+)+(\.png|\.jpg))'
AUDIO_REGEX = r'(?P<path>(/[\w_-]+)+(\.wav|\.mp3))'
LINK_REGEX = r'`*\!?\[(?P<alt>.+)\]\((sandbox:)?(file://)?{}\)`*'


def file_base64(file):
    bytes = Path(file).read_bytes()
    return base64.b64encode(bytes).decode()


def img_to_html(file, height):
    base64 = file_base64(file)
    # return f'<img src="data:image/png;base64,{base64}" style="height:{height}px;display:block;margin-left:auto;margin-right:auto">'  # align at center
    return f'<img src="data:image/png;base64,{base64}" style="height:{height}px;display:block;margin-right:auto">'


def audio_to_html(file):
    base64 = file_base64(file)
    # return f'<audio controls="controls" style="display:block;margin-left:auto;margin-right:auto"><source src="data:audio/wav;base64,{base64}"></audio>'  # align at center
    return f'<audio controls="controls" style="display:block;margin-right:auto"><source src="data:audio/wav;base64,{base64}"></audio>'


def process_answer(answer: str) -> str:
    images = list(re.finditer(LINK_REGEX.format(IMG_REGEX), answer))

    for item in reversed(images):
        before = answer[:item.start()]
        after = answer[item.end():]
        path = Path(item.groupdict()['path'])

        if path.exists():
            answer = f'{before}\n\n{img_to_html(path, 350)}\n\n{after}'
        else:
            answer = before + after

    images = list(re.finditer(IMG_REGEX, answer))

    for item in reversed(images):
        before = answer[:item.start()]
        after = answer[item.end():]
        path = Path(item.groupdict()['path'])

        if path.exists():
            answer = f'{before}\n\n{img_to_html(path, 350)}\n\n{after}'
        else:
            answer = before + after

    audios = list(re.finditer(LINK_REGEX.format(AUDIO_REGEX), answer))

    for item in reversed(audios):
        before = answer[:item.start()]
        after = answer[item.end():]
        path = Path(item.groupdict()['path'])

        if path.exists():
            answer = f'{before}\n\n{audio_to_html(path)}\n\n{after}'
        else:
            answer = before + after

    audios = list(re.finditer(AUDIO_REGEX, answer))

    for item in reversed(audios):
        before = answer[:item.start()]
        after = answer[item.end():]
        path = Path(item.groupdict()['path'])

        if path.exists():
            answer = f'{before}\n\n{audio_to_html(path)}\n{after}'
        else:
            answer = before + after

    return answer


def process_result_text(text: str) -> str:
    images = list(re.finditer(IMG_REGEX, text))

    for item in reversed(images):
        before = text[:item.start()]
        after = text[item.end():]
        text = f'{before}image{after}'

    audios = list(re.finditer(AUDIO_REGEX, text))

    for item in reversed(audios):
        before = text[:item.start()]
        after = text[item.end():]
        text = f'{before}audio{after}'

    return text


rootdir = Path(__file__).absolute().parents[1]
tmpdir = Path('.tmp')
state = st.session_state
icon_base64 = file_base64(rootdir / 'docs/src/agentlego-logo.png')

KEY_HTML = """\
<p style='text-align: left;display:flex;'>
  <span style='font-size:14px;font-weight:600;width:30px;text-align:justify;text-justify:inter-character;margin-right:5px;'>
    {key}
  </span>
  <span style='width:14px;text-align:left;display:block;'>:</span>
  <span style='flex:1;'>
    {value}
  </span>
</p>
"""

IMG_SUFFIX = ('.png', '.jpg', '.jpeg')
AUDIO_SUFFIX = ('.mp3', '.wav')
examples = [
    dict(
        title=':national_park: Draw similar image in different style.',
        user=('Please describe the above image, '
              'and draw a similar image in anime style.'),
        files=[load_image(rootdir / 'examples/demo.png')],
        tools=['VQA', 'TextToImage'],
    )
]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--tools',
    nargs='+',
    default=['Calculator'],
    choices=list_tools(),
    help='Add one or more animals of your choice')
args = parser.parse_args()


@st.cache_resource
def init_tmpdir():
    from shutil import rmtree
    rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(exist_ok=True)


@st.cache_resource
def init_tools():
    tools = [load_tool(name, name=name) for name in args.tools]
    for tool in tools:
        logger.info(f'Loading {tool.name}')
        tool.setup()
        tool._is_setup = True
    return {tool.name: tool for tool in tools}


def init_session(tools):
    """Initialize session state variables."""

    state.setdefault('history', [])
    state.setdefault('model_map', {})
    state.setdefault('files', [])
    state.setdefault('tools_selected', tools.keys())
    state.setdefault('chatbot', None)
    state.setdefault('disable_chat', False)
    state.setdefault('disable_clear', False)
    state.setdefault('disable_select', False)

    state.setdefault('example_user_input', None)
    state.setdefault('uuid', str(uuid.uuid4()))


def clear_session():
    """Clear the existing session state."""
    state.history = []
    state.files = state.file_uploader
    state.tools_selected = state.multiselect_tools
    state.disable_chat = False
    state.disable_clear = False
    state.disable_select = False
    state.chatbot = None
    state.example_user_input = None
    state.setdefault('uuid', str(uuid.uuid4()))


def get_model(model_name):
    """Initialize the model based on the selected option."""
    model = state.model_map.get(model_name)

    if model is None:
        from lagent.llms import GPTAPI
        if model_name == 'gpt-4-turbo':
            model_name = 'gpt-4-1106-preview'
        model = GPTAPI(
            model_type=model_name,
            temperature=0.8,
            key=state.api_key,
        )
        state.model_map[model_name] = model
    return model


def init_chatbot(model_name, tools):
    logger.info(f'Init {model_name} with: ' + ', '.join([tool.name for tool in tools]))
    chatbot = ReAct(
        llm=get_model(model_name),
        action_executor=ActionExecutor(actions=[tool.to_lagent() for tool in tools]),
        protocol=ReActProtocol(),
        max_turn=10,
    )
    return chatbot


class StreamlitUI:

    @staticmethod
    def init_page():
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout='wide', page_title='AgentLego-web', page_icon='icon.png')
        st.markdown(
            f'<h1><img src="data:image/png;base64,{icon_base64}" height=60px>'
            '<span style="color: #564bff;">AgentLego</span> Demo</hi>',
            unsafe_allow_html=True)

    @staticmethod
    def setup_sidebar(tools):
        """Setup the sidebar for model and plugin selection."""

        st.sidebar.selectbox(
            'Model',
            options=[
                'gpt-4-turbo',
                'gpt-3.5-turbo',
            ],
            key='model_name',
            disabled=state.disable_select,
            on_change=clear_session,
        )

        def update_api_key():
            model = state.model_map.get(state.model_name)
            if model is None or 'gpt' not in state.model_name:
                return
            model.keys = [state.api_key]

        st.sidebar.text_input('API key', key='api_key', on_change=update_api_key)

        st.sidebar.multiselect(
            'Tools',
            options=list(tools),
            default=tools.keys(),
            key='multiselect_tools',
            disabled=state.disable_select,
            on_change=clear_session,
        )

        st.sidebar.button(
            'Clear',
            on_click=clear_session,
            disabled=state.disable_clear,
            use_container_width=True,
        )

        def update_files():
            state.files = state.file_uploader

        st.sidebar.file_uploader(
            'Upload file',
            type=['png', 'jpg', 'jpeg', 'mp3', 'wav'],
            accept_multiple_files=True,
            key='file_uploader',
            on_change=update_files,
        )

        st.sidebar.markdown(
            '<hr><div style="text-align:center;font-size:1.2em;margin-bottom:1em">Try it!<div>',
            unsafe_allow_html=True)

        def click_example(example: dict):
            clear_session()
            files = []
            for file in example['files']:
                from streamlit.runtime.uploaded_file_manager import UploadedFile
                files.append(UploadedFile(file, None))
            state.files = files
            for tool in example['tools']:
                if tool not in tools:
                    st.warning(f'The tool `{tool}` is not loaded.')
                    return
            state.tools_selected = example['tools']
            state.disable_select = True
            state.disable_clear = True
            state.disable_chat = True
            state.example_user_input = example['user']

        with st.sidebar:
            for example in examples:
                st.button(
                    example['title'],
                    use_container_width=True,
                    on_click=partial(click_example, example))

    @staticmethod
    def render_user(prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    @staticmethod
    def render_assistant(content):
        with st.chat_message('assistant'):
            for response in content:
                if isinstance(response, Exception):
                    st.error(repr(response))
                else:
                    StreamlitUI.render_action(response)

    @staticmethod
    def add_item(key, inline_text=''):
        html = KEY_HTML.format(key=key, value=inline_text)
        st.markdown(html, unsafe_allow_html=True)

    @staticmethod
    def render_action(action: ActionReturn, retry: Tuple[int, int] = None):
        if action.type == 'FinishAction':
            reply = action.result['text']

            st.markdown(process_answer(reply), unsafe_allow_html=True)
            return

        if action.type in ['NoAction', 'InvalidAction']:
            if retry is not None:
                label = f':red[Fail (retry {retry[0]}/{retry[1]})]'
            else:
                label = ':red[Fail]'

            with st.expander(label):
                st.markdown(action.thought)
            return

        if retry is not None:
            action_type = action.type
            label = f':red[{action_type} (retry {retry[0]}/{retry[1]})]'
        else:
            label = action.type

        with st.expander(label):
            StreamlitUI.add_item('Thought', action.thought)
            args = []

            for k, v in action.args.items():
                if isinstance(v, str) and v.endswith(IMG_SUFFIX):
                    args.append(f'{k}=image')
                elif isinstance(v, str) and v.endswith(AUDIO_SUFFIX):
                    args.append(f'{k}=audio')
                else:
                    args.append(f'{k}={repr(v)}')

            st.markdown(
                "<p style='text-align: left;display:flex;'> "
                "<span style='font-size:14px;font-weight:600;'>"
                f'<img src="data:image/png;base64,{icon_base64}" height=20px> {action.type}({", ".join(args)})'
                '</span>',
                unsafe_allow_html=True)

            if isinstance(action.result, dict):
                st.markdown(process_result_text(action.result['text']))
                for image in action.result.get('image', []):
                    w, h = Image.open(image).size
                    st.image(image, caption='Generated Image', width=int(350 / h * w))
                for audio in action.result.get('audio', []):
                    st.audio(audio)
            elif action.errmsg:
                st.error(action.errmsg)


def main():
    # Initialize webpage
    StreamlitUI.init_page()
    init_tmpdir()

    # Initialize tools (run only once when start the streamlit app)
    tools = init_tools()

    init_session(tools)
    StreamlitUI.setup_sidebar(tools)

    # Create chatbot
    if state.chatbot is None:
        chatbot = init_chatbot(
            state.model_name,
            [tools[name] for name in state.tools_selected],
        )
        state.chatbot = chatbot
        state.system = chatbot._protocol.call_protocol

    # Display uploaded files
    files = []
    for file in state.files:
        file.seek(0)
        file_bytes = file.read()
        file_hash = hashlib.sha256(file_bytes[:512] + file_bytes[-512:])
        date = datetime.now().strftime('%Y%m%d-')
        file_name = date + file_hash.hexdigest()[:8] + Path(file.name).suffix

        # Save the file to a temporary location and get the path
        file_path = tmpdir / file_name
        if not file_path.exists():
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)

        if 'image' in file.type:
            w, h = Image.open(file).size
            st.image(
                file_bytes,
                caption='Uploaded Image',
                width=int(250 / h * w),
            )
            files.append(f'an image at `{file_path}`')
        elif 'audio' in file.type:
            st.audio(file_bytes, caption='Uploaded Audio')
            files.append(f'an audio at `{file_path}`')
    if files:
        file_prompt = f'\nUser have uploaded {len(files)} files: ' + ', '.join(
            files) + '.'
    else:
        file_prompt = ''
    extra_prompt = '\nThe Final Answer should be in the same language as the user input\nYou can display image or audio by file path in the final answer.'
    date_prompt = f'\nDate of today is {datetime.now().strftime("%Y/%m/%d")}.'
    state.chatbot._protocol.call_protocol = (
        state.system + extra_prompt + date_prompt + file_prompt)

    # Display chat history
    for dialog in state.history:
        if dialog['role'] == 'user':
            StreamlitUI.render_user(dialog['content'])
        elif dialog['role'] == 'assistant':
            StreamlitUI.render_assistant(dialog['content'])

    def start_chat():
        state.disable_select = True
        state.disable_clear = True
        state.disable_chat = True

    user = state.example_user_input or st.chat_input(
        '', on_submit=start_chat, disabled=state.disable_chat)
    if user:
        state.example_user_input = None
        StreamlitUI.render_user(user)
        state.history.append(dict(role='user', content=user))

        logger.info(
            f'UUID: {state.uuid}\nTools:{state.tools_selected}{file_prompt}\nUser: {user}'
        )
        responses = []
        with st.chat_message('assistant'):
            try:
                with st.spinner('Thinking'):
                    response: AgentReturn = state.chatbot.chat(user)
                for action in response.actions:
                    StreamlitUI.render_action(action)
                    responses.append(action)
                    logger.info(f'Action:\n{action}')
            except Exception as e:
                responses.append(e)
                st.error(repr(e))
                logger.info(f'Error: {repr(e)}')

        state.history.append(dict(role='assistant', content=copy.deepcopy(responses)))
        state.disable_chat = False
        state.disable_clear = False
        st.rerun()


if __name__ == '__main__':
    main()
