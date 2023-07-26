from unittest import skipIf

from mmengine import is_installed

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestInstructPix2PixTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('InstructPix2PixTool', device='cuda')
        img_path = 'tests/data/images/dog-image.jpg'
        res = tool(f'{img_path}, watercolor painting')
        assert isinstance(res, str)
