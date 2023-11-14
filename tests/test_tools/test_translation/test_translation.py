import pytest

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
    tool = tool.to_transformers_agent()
    hf_agent.prepare_for_new_chat()
    hf_agent._toolbox = {tool.name: tool}

    out = hf_agent.chat(f'Please translate the `{text}` from {source_lang} '
                        f'to {target_lang}.`')
    assert out.startswith('Les légumes')


def test_lagent(tool, lagent_agent):
    lagent_agent.new_session([tool.to_lagent()])

    out = lagent_agent.chat(
        f'Translate the `{text}` from {source_lang} to {target_lang}')
    assert out.actions[-1].valid == 1
    assert out.response.startswith('Les légumes')
