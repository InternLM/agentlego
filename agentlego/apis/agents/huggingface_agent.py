# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from transformers.tools import Tool

from agentlego.parsers import HuggingFaceAgentParser
from agentlego.tools.base import BaseTool
from ..tool import list_tools, load_tool


class HFAgentTool(Tool):
    """Adapter for agentlego.tools.Tool to transformers.tools.Tool."""

    def __init__(self, tool: BaseTool):
        self.tool = tool

        # remove spaces in the tool name which is not allowed in the hugging
        # face agent system
        self.name: str = 'agentlego_' + tool.name.lower().replace(' ', '_')
        self.description: str = tool.description
        self.inputs: List = list(tool.toolmeta.inputs)
        self.outputs: List = list(tool.toolmeta.outputs)

    def __call__(self, *args, **kwargs):
        return self.tool(*args, **kwargs)


def load_tools_for_hfagent(
    tools: Optional[List[Union[BaseTool, str]]] = None,
    device: str = 'cpu',
) -> List[HFAgentTool]:
    """Load a set of tools and adapt them to the hugginface agent tool
    interface.

    Args:
        tools (List[BaseTool, str] | None): A list of tool names or tools.
            If None, construct all available tools. Defaults to None.
        device (str): The device to load tools. If ``tools`` is a list of
            tool instances, it won't be used. Defaults to 'cpu'.

    Returns:
        list(HFAgentTool): loaded tools
    """
    tools = tools if tools is not None else list_tools()

    loaded_tools = []
    for tool in tools:
        if isinstance(tool, str):
            tool = load_tool(
                tool, device=device, parser=HuggingFaceAgentParser)
        else:
            tool.set_parser(HuggingFaceAgentParser)
        loaded_tools.append(HFAgentTool(tool))

    return loaded_tools
