from unittest import skipIf

from mmengine import is_installed

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, LangChainParser


@skipIf(not is_installed('transformers'), reason='requires transformers')
class TestTranslation(ToolTestCase):

    def test_call(self):
        text = 'Legumes share resources with nitrogen-fixing bacteria'
        source_lang = 'English'
        target_lang = 'French'

        tool = load_tool(
            'Translation', parser=LangChainParser(), device='cuda')
        res = tool(text, source_lang, target_lang)
        assert isinstance(res, str)

        tool = load_tool(
            'Translation', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool(text, source_lang, target_lang)
        assert isinstance(res, str)
