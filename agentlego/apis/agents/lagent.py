# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import re
from collections import defaultdict
from typing import List, Optional, Union

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from agentlego.parsers import DefaultParser
from agentlego.tools.base import BaseTool
from ..tool import list_tools, load_tool


class LagentTool(BaseAction):
    """A wrapper to align with the interface of Lagent tools."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(DefaultParser)

        name = tool.name.replace(' ', '')
        example_args = ', '.join(f'"{item}": xxx'
                                 for item in tool.input_fields)
        description = (f'{tool.description} Combine all args to one json '
                       f'string like {{{example_args}}}')

        self.tool = tool
        super().__init__(
            description=description,
            name=name,
            enable=True,
        )

    def run(self, json_args: str):
        # extract from json code block
        match_item = re.match(r'```(json)?(.+)```', json_args.strip(),
                              re.MULTILINE | re.DOTALL)
        if match_item is not None:
            json_args = match_item.group(2).strip()

        # load json format arguments
        try:
            kwargs = json.loads(json_args.strip(' .\'"\n`'))
        except Exception:
            error = ValueError(
                'All arguments should be combined into one json string.')
            return ActionReturn(
                type=self.name,
                errmsg=repr(error),
                state=ActionStatusCode.ARGS_ERROR,
                args={'raw_input': json_args},
            )

        try:
            result = self.tool(**kwargs)
            result_dict = defaultdict(list)
            result_dict['text'] = str(result)

            if not isinstance(result, tuple):
                result = [result]

            for res, out_type in zip(result, self.tool.toolmeta.outputs):
                if out_type != 'text':
                    result_dict[out_type].append(res)

            return ActionReturn(
                type=self.name,
                args=kwargs,
                result=result_dict,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=repr(e),
                args=kwargs,
                state=ActionStatusCode.API_ERROR,
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
            tool = load_tool(tool, device=device)
        loaded_tools.append(LagentTool(tool))

    return loaded_tools
