from unittest import skipIf

from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectReplace(ToolTestCase):

    def test_call(self):
        img_path = 'tests/data/images/test.jpg'
        tool = load_tool(
            'ObjectReplace', parser=VisualChatGPTParser(), device='cpu')
        res = tool(img_path, 'dog', 'a cartoon dog')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ObjectReplace', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, 'dog', 'a cartoon dog')
        assert isinstance(res, Image.Image)
