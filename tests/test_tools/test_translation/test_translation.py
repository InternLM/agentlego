import pytest

from agentlego.apis.agents import load_tools_for_hfagent, load_tools_for_lagent
from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool

text = 'Legumes share resources with nitrogen-fixing bacteria'
source_lang = 'English'
target_lang = 'French'


@pytest.fixture()
def tool():
    from agentlego.tools import Translation
    return setup_tool(Translation, device='cuda')


def test_call(tool):
    tool.set_parser(NaiveParser)
    res = tool(text, source_lang, target_lang)
    assert isinstance(res, str)


def test_hf_agent(tool, hf_agent):
    tools = load_tools_for_hfagent([tool])
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {t.name: t for t in tools}

    out = hf_agent.chat(f'Please translate the `{text}` from {source_lang} '
                        f'to {target_lang}.`')
    assert out.startswith('Les légumes')


def test_lagent(tool, lagent_agent):
    tools = load_tools_for_lagent([tool])
    lagent_agent.new_session(tools)

    out = lagent_agent.chat(
        f'Translate the `{text}` from {source_lang} to {target_lang}')
    assert out.actions[-1].valid == 1
    assert out.response.startswith('Les légumes')
