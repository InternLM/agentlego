import pytest

import mmlmtools.types as types
from mmlmtools.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from mmlmtools.parsers import NaiveParser
from mmlmtools.testing import setup_tool


@pytest.fixture()
def tool():
    from mmlmtools.tools import TextToImage
    return setup_tool(TextToImage, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool('generate an image of a cat')
    assert isinstance(res, types.ImageIO)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat('generate an image of a cat')
    assert out is not None


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat('generate an image of a cat')
    assert out.actions[-1].valid == 1
    assert '.png' in out.response
