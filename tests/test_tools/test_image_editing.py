from unittest import skipIf

from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectReplaceTool(ToolTestCase):

    def test_call(self):
        img_path = 'tests/data/images/test-image.jpeg'
        tool = load_tool(
            'ObjectReplaceTool', parser=VisualChatGPTParser(), device='cpu')
        res = tool(img_path, 'dog', 'a cartoon dog')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ObjectReplaceTool', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, 'dog', 'a cartoon dog')
        assert isinstance(res, Image.Image)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectRemoveTool(ToolTestCase):

    def test_call(self):
        img_path = 'tests/data/images/test-image.jpeg'
        tool = load_tool(
            'ObjectRemoveTool', parser=VisualChatGPTParser(), device='cpu')
        res = tool(img_path, 'dog')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'ObjectRemoveTool', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, 'dog')
        assert isinstance(res, Image.Image)
