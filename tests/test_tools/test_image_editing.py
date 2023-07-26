from unittest import skipIf

from mmengine import is_installed

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectReplaceTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('ObjectReplaceTool', device='cpu')
        res = tool('tests/data/images/test-image.jpeg, dog, lush green field')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectRemoveTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('ObjectRemoveTool', device='cpu')
        res = tool('tests/data/images/test-image.jpeg, dog')
        assert isinstance(res, str)
