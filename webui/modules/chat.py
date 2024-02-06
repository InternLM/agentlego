import copy
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import gradio as gr
from pydantic_core import to_jsonable_python

from . import shared
from .html_generator import chat_html_wrapper, reply_to_html
from .logging import logger
from .text_generation import generate_reply
from .utils import delete_file

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a')


def persist_file(path):
    path = Path(path)
    with open(path, 'rb') as file:
        file_bytes = file.read()
        file_hash = hashlib.sha256(file_bytes[:512] + file_bytes[-512:])
        date = datetime.now().strftime('%Y%m%d-')
        filename = date + file_hash.hexdigest()[:8] + path.suffix
        new_path = Path('generated/upload/')
        new_path.mkdir(parents=True, exist_ok=True)
        new_path = new_path / filename
    if not new_path.exists():
        path.rename(new_path)
    return new_path


def add_file(path):
    path = Path(path)
    if not path.exists():
        return '', ''

    if path.suffix in IMAGE_EXTENSIONS:
        internal = f'An image at `{path}`.\n'
        visible = f'<div><img src="file/{str(path)}" alt="Image"></div>'
    elif path.suffix in AUDIO_EXTENSIONS:
        internal = f'An audio at `{path}`.\n'
        visible = f'<div><audio controls="controls"><source src="file/{str(path)}"></audio></div>'
    else:
        internal = f'A file at `{path}\n'
        visible = f'<div>{path}</div>'

    return internal, visible


def chatbot_wrapper(text, state, files=None, regenerate=False, loading_message=True):
    history = state['history']
    if not text and not regenerate:
        yield state['history']
        return
    output = copy.deepcopy(history)
    just_started = True
    visible_text = None

    # Prepare the input
    if not regenerate:
        visible_text = html.escape(text)
        if files:
            for file in reversed(files):
                internal, visible = add_file(file)
                visible_text = visible + visible_text
                text = internal + text
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        output['visible'].pop()
        output['internal'].pop()

    # *Is typing...*
    if loading_message:
        yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(text, state, history=output)):

        visible_reply = reply_to_html(reply)

        if shared.stop_everything:
            yield output
            return

        if just_started:
            just_started = False
            output['internal'].append(['', []])
            output['visible'].append(['', ''])

        if not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            yield output

    yield output


def generate_chat_reply(text, state, file=None, regenerate=False, loading_message=True):
    history = state['history']
    if regenerate:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    for history in chatbot_wrapper(text, state, file, regenerate=regenerate, loading_message=loading_message):
        yield history


def generate_chat_reply_wrapper(text, state, file=None, regenerate=False):
    """Same as above but returns HTML for the UI."""
    for i, history in enumerate(generate_chat_reply(text, state, file, regenerate, loading_message=True)):
        yield chat_html_wrapper(history), history

def upload_file(file, uploaded, history):
    if not file:
        return gr.update()
    file = persist_file(file)
    uploaded.append(file)

    output = copy.deepcopy(history)
    visible = ''
    for file in uploaded:
        visible += add_file(file)[1]
    output['visible'] = output['visible'] + [[visible, '']]
    return uploaded, chat_html_wrapper(output)

def redraw_html(history):
    return chat_html_wrapper(history)


def modify_last_message(history):
    if len(history['visible']) > 0 and history['visible'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = history['internal'].pop()
        history['visible'].pop()
    else:
        last = ['', '']

    return last[0], history


def start_new_chat():
    history = {'internal': [], 'visible': []}

    from .message_schema import Answer
    from .settings import get_agent_settings
    if shared.agent_name:
        agent_settings = get_agent_settings(shared.agent_name)
        greeting = agent_settings.get('greeting', None)
        if greeting:
            history['internal'] += [['', [Answer(text=greeting)]]]
            history['visible'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]

    return history


def get_history_file_path(unique_id):
    p = Path(f'logs/{unique_id}.json')
    return p


def save_history(history, unique_id):
    if not unique_id:
        return

    p = get_history_file_path(unique_id)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)

    history = to_jsonable_python(history)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps(history, indent=4, ensure_ascii=False))


def rename_history(old_id, new_id):
    old_p = get_history_file_path(old_id)
    new_p = get_history_file_path(new_id)
    if new_p.parent != old_p.parent:
        logger.error(f'The following path is not allowed: {new_p}.')
    elif new_p == old_p:
        logger.info('The provided path is identical to the old one.')
    else:
        logger.info(f'Renaming {old_p} to {new_p}')
        old_p.rename(new_p)


def find_all_histories():
    paths = Path('logs/').glob('*.json')
    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
    histories = [path.stem for path in histories]

    return histories


def load_latest_history():
    """Loads the latest history for the given character in chat or chat-
    instruct mode, or the latest instruct history for instruct mode."""

    histories = find_all_histories()

    if len(histories) > 0:
        unique_id = Path(histories[0]).stem
        history = load_history(unique_id)
    else:
        history = start_new_chat()

    return history

def recover_message(data):
    from . import message_schema as msg

    if isinstance(data, dict) and '_role' in data:
        data_type = getattr(msg, data['_role'])
        return data_type.model_validate(data)
    elif isinstance(data, dict):
        return {k: recover_message(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return type(data)([recover_message(item) for item in data])
    else:
        return data

def load_history(unique_id):
    p = get_history_file_path(unique_id)

    f = json.loads(open(p, 'rb').read())
    return recover_message(f)


def delete_history(unique_id):
    p = get_history_file_path(unique_id)
    delete_file(p)
