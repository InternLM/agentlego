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
        'and outputs a {{{output:image}}}.')

    def apply(self, image: Image.Image, query: str) -> Image.Image:
        print(query)
        return image


class TestBaseToolv2(ToolTestCase):

    def test_call_with_visual_chatgpt(self):
        tool = DummyTool(parser=VisualChatGPTParser())

        print(tool.name)
        print(tool.input_types)
        print(tool.output_types)
        print(tool.description)

        with tempfile.TemporaryDirectory() as tempdir:
            path = osp.join(tempdir, 'image.jpg')
            img = np.random.randint(0, 255, [224, 224, 3]).astype(np.uint8)
            Image.fromarray(img).save(path)
            query = 'What is the color of the car?'

            inputs = f'{path}, {query}'
            outputs = tool(inputs)
            print(outputs)

    def test_call_with_huggingface_agent(self):
        tool = DummyTool(parser=HuggingFaceAgentParser())

        print(tool.name)
        print(tool.input_types)
        print(tool.output_types)
        print(tool.description)

        img = Image.fromarray(
            np.random.randint(0, 255, [224, 224, 3]).astype(np.uint8))
        query = 'What is the color of the car?'

        outputs = tool(img, query)
        print(type(outputs))
