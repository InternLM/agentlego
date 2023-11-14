from pathlib import Path

import pytest

from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool

data_dir = Path(__file__).parents[2] / 'data'
test_audio = (data_dir / 'audio/speech_to_text.flac').absolute()


@pytest.fixture()
def tool():
    from agentlego.tools import SpeechToText
    return setup_tool(SpeechToText, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool(str(test_audio))
    assert isinstance(res, str)


def test_hf_agent(tool, hf_agent):
    tool = tool.to_transformers_agent()
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {tool.name: tool}

    out = hf_agent.chat(f'Convert the audio `{test_audio}` to text.')
    assert 'going along slushy country' in out


def test_lagent(tool, lagent_agent):
    lagent_agent.new_session([tool.to_lagent()])

    out = lagent_agent.chat(f'Convert the audio `{test_audio}` to text.')
    assert out.actions[-1].valid == 1
    assert 'going along slushy country' in out.response
