from pathlib import Path

import pytest

from mmlmtools.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from mmlmtools.parsers import NaiveParser
from mmlmtools.testing import setup_tool

data_dir = Path(__file__).parents[2] / 'data'
test_audio = (data_dir / 'audio/speech_to_text.flac').absolute()


@pytest.fixture()
def tool():
    from mmlmtools.tools import SpeechToText
    return setup_tool(SpeechToText, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool(str(test_audio))
    assert isinstance(res, str)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat(f'Convert the audio `{test_audio}` to text.')
    assert 'going along slushy country' in out


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat(f'Convert the audio `{test_audio}` to text.')
    assert out.actions[-1].valid == 1
    assert 'going along slushy country' in out.response
