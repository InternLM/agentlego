# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from agentlego.parsers.custom_parsers import LagentParser
from agentlego.tools.base import BaseTool
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
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=repr(e),
                args=args,
                state=ActionStatusCode.ARGS_ERROR,
            )


def load_tools_for_lagent(
    tools: Optional[List[Union[BaseTool, str]]] = None,
    device: str = 'cpu',
) -> List[LagentTool]:
    """Load a set of tools and adapt them to the Lagent tool interface.

    Args:
        tools (List[BaseTool, str] | None): A list of tool names or tools.
            If None, construct all available tools. Defaults to None.
        device (str): The device to load tools. If ``tools`` is a list of
            tool instances, it won't be used. Defaults to 'cpu'.

    Returns:
        List[LagentTool]: loaded tools.
    """
    tools = tools if tools is not None else list_tools()

    loaded_tools = []
    for tool in tools:
        if isinstance(tool, str):
            tool = load_tool(tool, device=device, parser=LagentParser)
        else:
            tool.set_parser(LagentParser)
        loaded_tools.append(LagentTool(tool))

    return loaded_tools
