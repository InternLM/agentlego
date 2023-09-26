from pathlib import Path

import pytest

from agentlego.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool
from agentlego.types import ImageIO

data_dir = Path(__file__).parents[2] / 'data'
test_image = (data_dir / 'images/dog.jpg').absolute()


@pytest.fixture()
def tool():
    from agentlego.tools import VisualQuestionAnswering
    return setup_tool(VisualQuestionAnswering, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    assert isinstance(tool(ImageIO(str(test_image)), 'prompt'), str)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat(f'Please describe the `{test_image}`')
    assert isinstance(out, str)


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat(f'Please describe the `{test_image}`')
    assert out.actions[-1].valid == 1
    assert isinstance(out.response, str)
