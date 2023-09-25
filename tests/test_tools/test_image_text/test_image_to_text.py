# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest

from agentlego.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool

data_dir = Path(__file__).parents[2] / 'data'
test_img = (data_dir / 'images/dog.jpg').absolute()


@pytest.fixture()
def tool():
    from agentlego.tools import ImageCaption
    return setup_tool(ImageCaption, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool(str(test_img))
    assert isinstance(res, str)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat(f'Please describe the image `{test_img}`.')
    assert out is not None


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat(f'Please describe the image `{test_img}`.')
    assert out.actions[-1].valid == 1
    assert 'dog' in out.response
