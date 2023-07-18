from unittest import skipIf

from mmengine import is_installed

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentAnything(ToolTestCase):

    def test_call(self):
        tool = load_tool('SegmentAnything', device='cpu')
        res = tool('tests/data/images/test-image.png')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentClicked(ToolTestCase):

    def test_call(self):
        tool = load_tool('SegmentClicked', device='cpu')
        res = tool('tests/data/images/test-image.png, '
                   'tests/data/images/test-mask.png')
        assert isinstance(res, str)
