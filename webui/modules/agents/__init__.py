import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from agentlego.utils import is_package_available
from ..logging import logger
from . import lagent_agent as lagent
from . import langchain_agent as langchain


@dataclass
class AgentCallbacks:
    load_llm: Callable
    cfg_widget: Callable
    generate: Callable


agent_func_map: Mapping[str, AgentCallbacks] = {
    'langchain.StructuredChat.ChatOpenAI':
    AgentCallbacks(
        load_llm=langchain.llm_chat_openai,
        cfg_widget=langchain.cfg_chat_openai,
        generate=langchain.generate_structured,
    ),
    'lagent.InternLM2Agent':
    AgentCallbacks(
        load_llm=lagent.llm_internlm2_lmdeploy,
        cfg_widget=lagent.cfg_internlm2,
        generate=lagent.generate_internlm2,
    )
}


def load_llm(agent_name):
    from ..settings import get_agent_settings
    logger.info(f'Loading {agent_name}...')
    agent_settings = get_agent_settings(agent_name)
    loader = agent_settings.pop('agent_class')
    output = agent_func_map[loader].load_llm(agent_settings)
    return output


def clear_cache():
    gc.collect()
    if is_package_available('torch'):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_agent():
    from .. import shared
    shared.agent = None
    clear_cache()

def delete_agent(name):
    from .. import shared

    name = name.strip()

    if name == '':
        return
    if name == shared.agent_name:
        unload_agent()
        shared.agent_name = None

    p = Path(shared.args.agent_config)
    if p.exists():
        settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        settings = {}

    settings.pop(name, None)
    shared.agent_settings = settings

    output = yaml.dump(settings, sort_keys=False, allow_unicode=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(output)

    return f'`{name}` is deleted from `{p}`.'
