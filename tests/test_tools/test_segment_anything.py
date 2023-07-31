from unittest import skipIf

from mmengine import is_installed

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import VisualChatGPTParser


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentAnything(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'SegmentAnything', parser=VisualChatGPTParser(), device='cpu')
        res = tool('tests/data/images/test-image.png')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentClicked(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'SegmentClicked', parser=VisualChatGPTParser(), device='cpu')
        res = tool('tests/data/images/test-image.png, '
                   'tests/data/images/test-mask.png')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectSegmenting(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'ObjectSegmenting', parser=VisualChatGPTParser(), device='cpu')
        res = tool('tests/data/images/test-image.png, '
                   'water cup')
        assert isinstance(res, str)
