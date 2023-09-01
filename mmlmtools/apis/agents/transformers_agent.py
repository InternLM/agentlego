# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from transformers.tools import Tool

from mmlmtools.parsers import HuggingFaceAgentParser
from mmlmtools.tools.base import BaseTool
from ..tool import list_tools, load_tool


class HFAgentTool(Tool):
    """Adapter for mmlmtools.tools.Tool to transformers.tools.Tool."""

    def __init__(self, tool: BaseTool):
        self.tool = tool

        self.name: str = tool.name
        self.description: str = tool.description
        self.inputs: List = list(tool.toolmeta.inputs)
        self.outputs: List = list(tool.toolmeta.outputs)

    def __call__(self, *args, **kwargs):
        return self.tool(*args, **kwargs)


def load_tools_for_hfagent(tool_names: Optional[List[str]] = None,
                           device: str = 'cpu') -> List[HFAgentTool]:
    """Load a set of tools and adapt them to the transformers agent tool
    interface.

    Args:
        tool_names (list[str]): list of tool names. Defaults to None, which
            means all tools will be loaded.
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
    list(HFAgentTool): loaded tools
    """
    tool_names = tool_names or list_tools()
    loaded_tools = []
    for name in tool_names:
        tool = load_tool(name, device=device, parser=HuggingFaceAgentParser)
        # remove spaces in the tool name which is not allowed in the hugging
        # face agent system
        name = 'mm_' + tool.name.lower().replace(' ', '_')
        tool.toolmeta.name = name
        loaded_tools.append(HFAgentTool(tool))

    return loaded_tools
