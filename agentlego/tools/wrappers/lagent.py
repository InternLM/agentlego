import copy
import json
import re
from collections import defaultdict

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from agentlego.parsers import DefaultParser
from agentlego.types import AudioIO, ImageIO
from ..base import BaseTool


class LagentTool(BaseAction):
    """A wrapper to align with the interface of Lagent tools."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(DefaultParser)  # Use string input & output
        self.tool = tool

        example_args = ', '.join(f'"{name}": xxx' for name in tool.arguments)
        description = (f'{tool.description} Combine all args to one json '
                       f'string like {{{example_args}}}')

        super().__init__(
            name=tool.name.replace(' ', ''),
            description=description,
            enable=True,
        )

    def run(self, json_args: str):
        # load json format arguments
        try:
            item = next(
                re.finditer('{.*}', json_args, re.MULTILINE | re.DOTALL))
            kwargs = json.loads(item.group())
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


class iLagentTool(BaseAction):
    """A wrapper to align with the interface of iLagent tools."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(DefaultParser)  # Use string input & output
        self.tool = tool

        super().__init__(
            name=tool.name.replace(' ', ''),
            description=self.tool.description,
            enable=True,
        )

    def run(self, **kwargs):
        try:
            result = self.tool(**kwargs)
            result_dict = {}
            result_dict['text'] = str(result)

            if not isinstance(result, tuple):
                result = [result]

            for out, p in zip(result, self.tool.outputs):
                if p.type is ImageIO:
                    result_dict['image'] = out
                elif p.type is AudioIO:
                    result_dict['audio'] = out

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
