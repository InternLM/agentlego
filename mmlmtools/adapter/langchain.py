# Copyright (c) OpenMMLab. All rights reserved.
from langchain.agents.tools import Tool

from mmlmtools import list_tool, load_tool


def load_mmtools_for_langchain(load_dict):
    """Load mmtools into langchain style.

    Args:
        load_dict (dict): {tool_name: device}

    Returns:
        langchain_tools (list): list of mmtools
    """
    mmtool_list = list_tool()
    langchain_tools = []
    for tool_name, device in load_dict:
        if tool_name in mmtool_list:
            mmtool = load_tool(tool_name, device=device)
            tool = Tool(
                name=mmtool.toolmeta.name,
                description=mmtool.toolmeta.description,
                func=mmtool)
            langchain_tools.append(tool)
    return langchain_tools
