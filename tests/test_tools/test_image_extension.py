from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


class TestImageExtension(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'ImageExtension', parser=VisualChatGPTParser(), device='cpu')
        img_path = 'tests/data/images/test.jpg'
        res = tool(img_path, '2000x1000')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ImageExtension',
            parser=HuggingFaceAgentParser(),
            device='cpu')
        res = tool(img, '2000x1000')
        assert isinstance(res, Image.Image)
