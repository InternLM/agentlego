# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List, Union

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from mmlmtools.parsers.custom_parsers import LagentParser
from mmlmtools.tools.base import BaseTool
from ..tool import list_tools, load_tool


class LagentTool(BaseAction):
    """A wrapper to align with the interface of Lagent tools."""

    def __init__(self, tool: BaseTool):
        self.tool = tool

        super().__init__(
            description=tool.description,
            name=tool.name,
            enable=True,
        )

    def run(self, *args, **kwargs):
        try:
            result = self.tool(*args, **kwargs)
            return ActionReturn(
                type=self.name,
                args=args,
                result=dict(text=str(result)),
            )
        except json.JSONDecodeError:
            return ActionReturn(
                type=self.name,
                errmsg='The arguments should be format as a json string.',
                args=args,
                state=ActionStatusCode.ARGS_ERROR,
            )


def load_tools_for_lagent(tools: List[Union[BaseTool, str]],
                          device: str = 'cpu') -> List[LagentTool]:
    """Load a set of tools and adapt them to the transformers agent tool
    interface.

    Args:
        tool_names (list[str]): list of tool names
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
        List[LagentTool]: loaded tools.
    """
    default_tools = list_tools()
    loaded_tools = []

    for tool in tools:
        if isinstance(tool, str):
            assert tool in default_tools, f'{tool} is not a valid tool name.'
            tool = load_tool(tool, device=device, parser=LagentParser)
        else:
            tool.set_parser(LagentParser)
        loaded_tools.append(LagentTool(tool))

    return loaded_tools
