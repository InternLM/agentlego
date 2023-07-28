# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from transformers.tools import Tool

from mmlmtools.tools.base_tool import BaseTool
from mmlmtools.tools.parsers import TransformersAgentParser
from ..tools import list_tools, load_tool


class TFAgentTool(Tool):
    """Adapter for mmlmtools.tools.Tool to transformers.tools.Tool."""

    def __init__(self, tool: BaseTool):
        self.tool = tool

        self.name: str = tool.name
        self.description: str = tool.description
        self.inputs: List = list(tool.inputs)
        self.outputs: List = list(tool.outputs)

    def __call__(self, *args, **kwargs):
        return self.tool(*args, **kwargs)


def load_tools_for_tf_agent(tool_names: List[str],
                            device: str = 'cpu') -> List[TFAgentTool]:
    """Load a set of tools and adapt them to the transformers agent tool
    interface.

    Args:
        tool_names (list[str]): list of tool names
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
    list(TFAgentTool): loaded tools
    """
    all_tools = list_tools()
    loaded_tools = []
    for name in tool_names:
        if name not in all_tools:
            raise ValueError(f'{name} is not a valid tool name.')
        tool = load_tool(name, device=device, parser=TransformersAgentParser())
        loaded_tools.append(TFAgentTool(tool))

    return loaded_tools
