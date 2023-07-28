# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from langchain.agents.tools import Tool

from ..tools import list_tools, load_tool


def load_tools_for_langchain(tool_names: List[str], device: str = 'cpu'):
    """Load a set of tools and adapt them to the langchain tool interface.

    Args:
        tool_names (list[str]): list of tool names
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
    list(langchain.Tool): loaded tools
    """

    all_tools = list_tools()
    loaded_tools = []
    for name in tool_names:
        if name not in all_tools:
            raise ValueError(f'{name} is not a valid tool name.')
        tool = load_tool(name, device=device)
        langchain_tool = Tool(
            name=tool.toolmeta.name,
            description=tool.toolmeta.description,
            func=tool)
        loaded_tools.append(langchain_tool)
    return loaded_tools
