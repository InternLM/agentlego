from agentlego.apis.tool import list_tools, load_tool
from agentlego.tools import Calculator


def test_load_tool():
    # load tool without with default model
    tool = load_tool('Calculator')
    assert isinstance(tool, Calculator)

    # description will be overwrite
    tool = load_tool('ImageDescription', description='custom')
    assert 'custom' in tool.toolmeta.description

    # cached tool
    tool = load_tool('Calculator')
    cached_tool = load_tool('Calculator')
    assert cached_tool is tool


def test_list_tools():

    assert 'Calculator' in list_tools()
