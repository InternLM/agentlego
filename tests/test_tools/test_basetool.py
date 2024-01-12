from agentlego.schema import Annotated, Info
from agentlego.tools import BaseTool
from agentlego.types import ImageIO


class DummyTool(BaseTool):
    default_desc = 'This is a dummy tool.'

    def apply(
        self,
        image: ImageIO,
        query: Annotated[str, Info('The query.')],
    ) -> Annotated[ImageIO, Info('The result image.')]:
        return image


def test_lagent():
    from lagent.actions import BaseAction
    tool = DummyTool().to_lagent()
    assert isinstance(tool, BaseAction)

    expected_description = '''\
This is a dummy tool.
Args: image (path); query (str, The query.)
Returns: path (The result image.)
Combine all args to one json string like {"image": xxx, "query": xxx}
'''

    assert tool.name == 'DummyTool'
    assert tool.description == expected_description


def test_hf_agent():
    from transformers.tools import Tool
    tool = DummyTool().to_transformers_agent()
    assert isinstance(tool, Tool)

    expected_description = '''\
This is a dummy tool.
Args: image (image); query (str, The query.)
Returns: image (The result image.)
'''

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
