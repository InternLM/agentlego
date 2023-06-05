from transformers.tools import Tool
from mmlmtools import list_tool, load_tool


class ToolAdapter(Tool):
    def __init__(self, tool):
        self.name = tool.toolmeta.tool_name
        self.tool = tool
    
        if 'image_path' in tool.input_style:
            self.tool.input_style = self.tool.input_style.replace('image_path', 'pil image')

        if 'image_path' in tool.output_style:
            self.tool.output_style = self.tool.output_style.replace('image_path', 'pil image')

        self.tool.format_description()

        self.description = tool.toolmeta.description

    def __call__(self, *args, **kwargs):
        return self.tool(*args, **kwargs)


def load_tools_for_hf_agent():
    tools = []
    for tool_name in list_tool():
        mmtool = load_tool(tool_name)
        hf_tool = ToolAdapter(mmtool)
        tools.append(hf_tool)

    return tools
