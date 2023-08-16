from unittest import skipIf

from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestTextToImage(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'TextToImage', parser=VisualChatGPTParser(), device='cuda')
        res = tool('generate an image of a cat')
        assert isinstance(res, str)

        tool = load_tool(
            'TextToImage', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool('generate an image of a cat')
        assert isinstance(res, Image.Image)
