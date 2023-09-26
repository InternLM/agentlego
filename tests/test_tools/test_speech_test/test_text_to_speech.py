# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from agentlego.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool
from agentlego.types import AudioIO


@pytest.fixture()
def tool():
    from agentlego.tools import TextToSpeech
    return setup_tool(TextToSpeech, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool('Hello world')
    assert isinstance(res, AudioIO)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat('Please speak out the text `Hello world`.')
    assert out is not None


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat('Please speak out the text `Hello world`.')
    assert out.actions[-1].valid == 1
    assert 'wav' in out.response
