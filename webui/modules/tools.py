import ast
import inspect
from copy import deepcopy
from pathlib import Path

import gradio as gr
import yaml

from . import shared
from .settings import get_tool_settings, save_tool_settings


def parse_args(args_str):
    call = ast.parse(f'foo({args_str})').body[0].value
    kwargs = {}
    for keyword in call.keywords:
        k = keyword.arg
        v = ast.Expression(body=keyword.value)
        ast.fix_missing_locations(v)
        kwargs[k] = eval(compile(v, '', 'eval'))
    return kwargs


def load_tool(name=None):
    cfg = get_tool_settings(name)
    if not cfg['enable']:
        shared.toolkits.pop(name, None)
        return None

    try:
        tool = load_tool_from_cfg(cfg)
        tool.setup()
        tool._is_setup = True
        shared.toolkits[name] = tool
        return tool
    except Exception as e:
        save_tool_settings(
            tool_class=cfg['class'],
            name=cfg['name'],
            desc=cfg['description'],
            enable=False,
            device=cfg.get('device', None),
            args=cfg['args'],
            old_name=cfg['name'])
        raise gr.Error(f'Failed to load tool `{name}`, auto disabled.') from e


def load_tool_from_cfg(tool_cfg):
    tool_cfg = deepcopy(tool_cfg)
    tool_class = shared.tool_classes[tool_cfg.pop('class')]
    device = tool_cfg.pop('device', 'cpu')
    kwargs = parse_args(tool_cfg.pop('args'))
    from agentlego.tools.remote import RemoteTool

    if 'device' in inspect.signature(tool_class).parameters:
        tool = tool_class(device=device, **kwargs)
    elif tool_class is RemoteTool:
        tool = RemoteTool.from_url(**kwargs)
    else:
        tool = tool_class(**kwargs)

    tool.toolmeta.name = tool_cfg['name']
    tool.toolmeta.description = tool_cfg['description']

    return tool

def delete_tool(name):
    name = name.strip()

    if name == '':
        return

    p = Path(shared.args.tool_config)
    if p.exists():
        settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        settings = {}

    settings.pop(name, None)
    shared.tool_settings = settings
    shared.toolkits.pop(name, None)

    output = yaml.dump(settings, sort_keys=False, allow_unicode=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(output)

    return f'`{name}` is deleted from `{p}`.'
