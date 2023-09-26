from PIL import Image

from agentlego import load_tool
from agentlego.parsers import HuggingFaceAgentParser, LangChainParser
from agentlego.testing import ToolTestCase


class TestImageExpansion(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'ImageExpansion', parser=LangChainParser(), device='cpu')
        img_path = 'tests/data/images/dog2.jpg'
        res = tool(img_path, '2000x1000')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ImageExpansion', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, '2000x1000')
        assert isinstance(res, Image.Image)
