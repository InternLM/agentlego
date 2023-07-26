import os.path as osp
import tempfile

import numpy as np
from PIL import Image

from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.base_tool_v2 import BaseToolv2
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


class DummyTool(BaseToolv2):
    DEFAULT_TOOLMETA = dict(
        name='Dummy Tool',
        model=None,
        description='This is a dummy tool. '
        'It takes an {{{input:image}}} and a {{{input:text}}} as the inputs, '
        'and outputs an {{{output:image}}}.')

    def apply(self, image: Image.Image, query: str) -> Image.Image:
        return image


class TestBaseToolv2(ToolTestCase):

    def test_call_with_visual_chatgpt(self):
        tool = DummyTool(parser=VisualChatGPTParser())

        expected_description = 'This is a dummy tool. It takes an image ' \
            'represented by path and a text represented by string as the ' \
            'inputs, and outputs an image represented by path. ' \
            'Inputs should be separated by comma.'

        self.assertEqual(tool.name, 'Dummy Tool')
        self.assertEqual(tool.inputs, ('image', 'text'))
        self.assertEqual(tool.outputs, ('image', ))
        self.assertEqual(tool.description, expected_description)

        with tempfile.TemporaryDirectory() as tempdir:
            path = osp.join(tempdir, 'image.jpg')
            img = np.random.randint(0, 255, [224, 224, 3]).astype(np.uint8)
            Image.fromarray(img).save(path)
            query = 'What is the color of the car?'

            inputs = f'{path}, {query}'
            outputs = tool(inputs)
            self.assertIsInstance(outputs, str)

    def test_call_with_huggingface_agent(self):
        tool = DummyTool(parser=HuggingFaceAgentParser())

        expected_description = 'This is a dummy tool. It takes an image ' \
            'and a text as the inputs, and outputs an image.' \

        self.assertEqual(tool.name, 'Dummy Tool')
        self.assertEqual(tool.inputs, ('image', 'text'))
        self.assertEqual(tool.outputs, ('image', ))
        self.assertEqual(tool.description, expected_description)

        img = Image.fromarray(
            np.random.randint(0, 255, [224, 224, 3]).astype(np.uint8))
        query = 'What is the color of the car?'

        outputs = tool(img, query)
        self.assertIsInstance(outputs, Image.Image)
