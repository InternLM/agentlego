from unittest import skipIf

from mmengine import is_installed
from PIL import Image

from agentlego import load_tool
from agentlego.testing import ToolTestCase
from agentlego.tools.parsers import HuggingFaceAgentParser, LangChainParser


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectRemove(ToolTestCase):

    def test_call(self):
        img_path = 'tests/data/images/dog2.jpg'
        tool = load_tool(
            'ObjectRemove', parser=LangChainParser(), device='cpu')
        res = tool(img_path, 'dog')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ObjectRemove', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, 'dog')
        assert isinstance(res, Image.Image)
