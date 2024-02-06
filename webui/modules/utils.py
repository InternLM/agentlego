import functools
import re
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Mapping, Tuple, Union

from agentlego.schema import ToolMeta
from agentlego.types import AudioIO, File, ImageIO
from . import shared
from .logging import logger


def parse_inputs(toolmeta: ToolMeta, args: Union[str, tuple, dict]) -> Mapping[str, dict]:
    if not args:
        return {}
    params = {p.name: p for p in toolmeta.inputs}

    if len(params) > 1 and isinstance(args, str):
        try:
            args = literal_eval(args)
        except Exception:
            pass

    if isinstance(args, str):
        args = {toolmeta.inputs[0].name: args}
    elif isinstance(args, tuple):
        args = {name: args for name in params}

    parsed_args = {}
    for k, v in args.items():
        if k not in params:
            parsed_args[k] = dict(type='text', content=v)
        elif params[k].type is ImageIO:
            parsed_args[k] = dict(type='image', content=v)
        elif params[k].type is AudioIO:
            parsed_args[k] = dict(type='audio', content=v)
        elif params[k].type is File:
            parsed_args[k] = dict(type='file', content=v)
        else:
            parsed_args[k] = dict(type='text', content=v)

    return parsed_args


def parse_outputs(toolmeta: ToolMeta, outs: Union[str, tuple, dict]) -> Tuple[dict, ...]:
    if not outs:
        return ()

    if len(toolmeta.outputs) > 1 and isinstance(outs, str):
        try:
            outs = literal_eval(outs)
        except Exception:
            pass

    if isinstance(outs, str):
        outs = (outs, )
    elif isinstance(outs, dict):
        outs = tuple(outs.values())

    parsed_outs = []
    for p, out in zip(toolmeta.outputs, outs):
        if p.type is ImageIO:
            parsed_outs.append(dict(type='image', content=out))
        elif p.type is AudioIO:
            parsed_outs.append(dict(type='audio', content=out))
        elif p.type is File:
            parsed_outs.append(dict(type='file', content=out))
        else:
            parsed_outs.append(dict(type='text', content=out))

    return tuple(parsed_outs)


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
