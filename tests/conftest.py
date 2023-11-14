from pathlib import Path
from types import MethodType

import pytest


@pytest.fixture(scope='module')
def hf_agent():
    from transformers import HfAgent
    prompt_file = Path(__file__).parents[1] / 'examples/hf_demo_prompts.txt'
    agent = HfAgent(
        'https://api-inference.huggingface.co/models/bigcode/starcoder',
        chat_prompt_template=prompt_file.read_text())
    agent._toolbox = {}
    return agent


@pytest.fixture(scope='module')
def lagent_agent():
    from lagent import GPTAPI, ActionExecutor, ReAct

    agent = ReAct(
        llm=GPTAPI(model_type='gpt-4', temperature=0.),
        max_turn=3,
        action_executor=ActionExecutor(actions=[]),
    )

    def new_session(self: ReAct, actions):
        self._action_executor = ActionExecutor(actions)
        self._session_history = []

    setattr(agent, 'new_session', MethodType(new_session, agent))

    return agent
