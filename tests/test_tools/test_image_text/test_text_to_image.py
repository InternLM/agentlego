import pytest

import agentlego.types as types
from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool


@pytest.fixture()
def tool():
    from agentlego.tools import TextToImage
    return setup_tool(TextToImage, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool('generate an image of a cat')
    assert isinstance(res, types.ImageIO)


def test_hf_agent(tool, hf_agent):
    tool = tool.to_transformers_agent()
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {tool.name: tool}

    out = hf_agent.chat('generate an image of a cat')
    assert out is not None


def test_lagent(tool, lagent_agent):
    lagent_agent.new_session([tool.to_lagent()])

    out = lagent_agent.chat('generate an image of a cat')
    assert out.actions[-1].valid == 1
    assert '.png' in out.response
