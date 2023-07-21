from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


class TestImageExtensionTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('ImageExtensionTool', device='cpu')
        res = tool('tests/data/images/test-image.jpeg, 1000x1000')
        assert isinstance(res, str)
