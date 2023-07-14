from unittest import skipIf

from mmengine import is_installed

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('transformers'), reason='requires transformers')
class TestTextQATool(ToolTestCase):

    def test_call(self):
        tool = load_tool('TextQuestionAnsweringTool', device='cuda')
        inputs = ('Tom is a student. He is in class 1. He is 10 years old. '
                  'How old is Tom?')
        res = tool(inputs)
        self.assertIsInstance(res, str)
