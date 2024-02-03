from copy import deepcopy
from pathlib import Path

import gradio as gr
import yaml

from . import shared, ui


def get_agent_settings(name):
    return deepcopy(shared.agents_settings[name])


def get_tool_settings(name):
    return deepcopy(shared.tool_settings[name])


def apply_agent_settings(agent):
    '''
    UI: update the state variable with the agent settings
    '''
    state = deepcopy(ui.agent_elements)
    if agent is None or agent == 'New Agent':
        return list(state.values())
    agent_settings = get_agent_settings(agent)
    agent_class = agent_settings.pop('agent_class')
    state['agent_class'] = agent_class
    state.update({f'{agent_class}#{k}': v for k, v in agent_settings.items()})
    return [state[k] for k in ui.agent_elements]


def apply_tool_settings_to_state(tool, state):
    '''
    UI: update the state variable with the tool settings
    '''
    if tool is None or tool == 'New Tool':
        state.update(dict(tool_class=None))
        return state
    tool_settings = get_tool_settings(tool)
    for k, v in tool_settings.items():
        state['tool_' + k] = v
    return state


def save_agent_settings(name, *args):
    """Save the settings for this agent to agent_config.yaml."""
    name = name.strip()

    if name == '':
        raise gr.Error('Please specify the agent name to save.')

    p = Path(shared.args.agent_config)
    if p.exists():
        settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        settings = {}

    settings.setdefault(name, {})

    state = {k: v for k, v in zip(ui.agent_elements, args)}
    for k in ui.agent_elements:
        if k == 'agent_class' or k.startswith(state['agent_class'] + '#'):
            save_k = k.rpartition('#')[-1]
            settings[name][save_k] = state[k] or None

    shared.agents_settings = settings

    output = yaml.dump(settings, sort_keys=False, allow_unicode=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(output)

    return f'Settings for `{name}` saved to `{p}`.'

def save_tool_settings(tool_class, name, desc, enable, device, args, old_name=None):
    """Save the settings for this agent to agent_config.yaml."""
    name = name.strip()

    if name == '':
        return 'Not saving the settings because no model is loaded.'

    p = Path(shared.args.tool_config)
    if p.exists():
        settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        settings = {}

    if old_name is not None:
        settings.pop(old_name, None)
    elif name in settings:
        return f'The name `{name}` is already used.'

    settings[name] = {
        'class': tool_class,
        'name': name,
        'description': desc,
        'enable': enable,
        'device': device,
        'args': args,
    }
    shared.tool_settings = settings

    output = yaml.dump(settings, sort_keys=False, allow_unicode=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(output)

    return f'Settings for `{name}` saved to `{p}`.'


def make_agent_params_visible(agent_class):
    updates = []
    for name in ui.agent_elements:
        if name == 'agent_class':
            updates.append(gr.update())
        elif not agent_class:
            updates.append(gr.update(visible=False, interactive=False))
        elif name.startswith(agent_class + '#'):
            updates.append(gr.update(visible=True, interactive=True))
        else:
            updates.append(gr.update(visible=False, interactive=False))

    return updates
