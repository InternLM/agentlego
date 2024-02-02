from agentlego.tools import BaseTool
from agentlego.types import Annotated, ImageIO, Info


class DummyTool(BaseTool):
    default_desc = 'This is a dummy tool.'

    def apply(
        self,
        image: ImageIO,
        query: Annotated[str, Info('The query')],
        option: bool = True,
    ) -> Annotated[ImageIO, Info('The result image')]:
        return image


def test_lagent():
    from lagent.actions import BaseAction
    tool = DummyTool().to_lagent()
    assert isinstance(tool, BaseAction)

    assert tool.name == 'DummyTool'
    assert tool.description['description'] == 'This is a dummy tool.'
    assert tool.description['required'] == ['image', 'query']
    assert tool.description['parameters'][1]['description'] == 'The query'


def test_hf_agent():
    from transformers.tools import Tool
    tool = DummyTool().to_transformers_agent()
    assert isinstance(tool, Tool)

    expected_description = '''\
This is a dummy tool.
Args: image (image); query (str, The query); option (bool. Optional, Defaults to True)
Returns: image (The result image)'''

    assert tool.name == 'agentlego_dummytool'
    assert tool.description == expected_description


def test_langchain():
    from langchain.tools import StructuredTool
    tool = DummyTool().to_langchain()
    assert isinstance(tool, StructuredTool)

    expected_description = '''\
DummyTool(image: str, query: str, option: str = True) - This is a dummy tool.
Args: image (path); query (str, The query); option (bool. Optional, Defaults to True)
Returns: path (The result image)'''

    assert tool.name == 'DummyTool'
    assert tool.description == expected_description
