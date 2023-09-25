from unittest import skipIf

from mmengine import is_installed

from agentlego import load_tool
from agentlego.testing import ToolTestCase
from agentlego.tools.parsers import LangChainParser


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentAnything(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'SegmentAnything', parser=LangChainParser(), device='cpu')
        res = tool('tests/data/images/cups.png')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestSegmentClicked(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'SegmentClicked', parser=LangChainParser(), device='cpu')
        res = tool('tests/data/images/cups.png, '
                   'tests/data/images/cups_mask.png')
        assert isinstance(res, str)


@skipIf(
    not is_installed('segment_anything'), reason='requires segment_anything')
class TestObjectSegmenting(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'ObjectSegmenting', parser=LangChainParser(), device='cpu')
        res = tool('tests/data/images/cups.png, water cup')
        assert isinstance(res, str)
