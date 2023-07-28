# Copyright (c) OpenMMLab. All rights reserved.
import weakref
from typing import List

from mmlmtools.tools.base_tool import BaseTool
from mmlmtools.tools.parsers import VisualChatGPTParser
from ..tools import list_tools, load_tool


class _Inference:

    def __init__(self, tool: BaseTool):
        self.tool = weakref.ref(tool)

    @property
    def name(self):
        return self.tool().name

    @property
    def description(self):
        return self.tool().description

    def __call__(self, *args, **kwargs):
        return self.tool()(*args, **kwargs)


def load_tools_for_visual_chatgpt(tool_names: List[str], device: str = 'cpu'):
    """Load a set of tools and adapt them to Visual ChatGPT style.

    Args:
        tool_names (list[str]): list of tool names
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
    list(Tool): loaded tools
    """
    all_tools = list_tools()
    loaded_tools = []
    for name in tool_names:
        if name not in all_tools:
            raise ValueError(f'{name} is not a valid tool name.')
        tool = load_tool(name, device=device, parser=VisualChatGPTParser())
        setattr(tool, 'inference', _Inference(tool))
        loaded_tools.append(tool)

    return loaded_tools
