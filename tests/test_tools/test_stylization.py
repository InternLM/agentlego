from unittest import skipIf

from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestInstructPix2Pix(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'InstructPix2Pix', parser=VisualChatGPTParser(), device='cuda')
        img_path = 'tests/data/images/dog-image.jpg'
        res = tool(f'{img_path}, watercolor painting')
        assert isinstance(res, str)

        img = Image.open(img_path)
        tool = load_tool(
            'InstructPix2Pix', parser=HuggingFaceAgentParser(), device='cpu')
        res = tool(img, 'watercolor painting')
        assert isinstance(res, Image.Image)
