from agentlego.parsers import DefaultParser
from agentlego.tools.base import BaseTool
from agentlego.types import ImageIO


class DummyTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Dummy Tool',
        description='This is a dummy tool. It takes an image and a text '
        'as the inputs, and returns an image.',
        inputs=('image', 'text'),
        outputs=('image', ),
    )

    def __init__(self):
        super().__init__(toolmeta=self.DEFAULT_TOOLMETA, parser=DefaultParser)

    def apply(self, image: ImageIO, query: str) -> ImageIO:
        return image


def test_lagent():
    from lagent.actions import BaseAction
    tool = DummyTool().to_lagent()
    assert isinstance(tool, BaseAction)

    expected_description = (
        'This is a dummy tool. It takes an image and a text as the inputs, '
        'and returns an image. Args: image (image path), query (text string) '
        'Combine all args to one json string like {"image": xxx, "query": xxx}'
    )

    assert tool.name == 'DummyTool'
    assert tool.description == expected_description


def test_hf_agent():
    from transformers.tools import Tool
    tool = DummyTool().to_transformers_agent()
    assert isinstance(tool, Tool)

    expected_description = (
        'This is a dummy tool. It takes an image and a text as '
        'the inputs, and returns an image. Args: image (image), query (text)')

    assert tool.name == 'agentlego_dummy_tool'
    assert tool.description == expected_description


def test_langchain():
    from langchain.tools import StructuredTool
    tool = DummyTool().to_langchain()
    assert isinstance(tool, StructuredTool)

    expected_description = (
        'Dummy Tool(image: str, query: str) - This is a dummy tool. '
        'It takes an image and a text as the inputs, and returns an image.')

    assert tool.name == 'Dummy Tool'
    assert tool.description == expected_description
