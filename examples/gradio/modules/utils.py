import functools
import re
from datetime import datetime
from pathlib import Path

from . import shared
from .logging import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and isinstance(keys[0], (list, tuple, set)):
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    with open(abs_path, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved {abs_path}.')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    if abs_path.exists():
        abs_path.unlink()
        logger.info(f'Deleted {fname}.')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_agents():
    return ['New Agent'] + sorted(shared.agents_settings.keys(), key=natural_keys)


def get_available_tools():
    return ['New Tool'] + sorted(shared.tool_settings.keys(), key=natural_keys)


@functools.lru_cache()
def get_available_devices():
    devices = ['cpu']
    try:
        import torch
        if torch.cuda.is_available():
            devices += [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return devices
