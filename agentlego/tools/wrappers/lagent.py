import copy

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from agentlego.parsers import DefaultParser
from agentlego.types import AudioIO, File, ImageIO
from ..base import BaseTool


def convert_type(t):
    if t in [str, ImageIO, AudioIO, File]:
        return 'STRING'
    elif t is int:
        return 'NUMBER'
    elif t is float:
        return 'FLOAT'
    elif t is bool:
        return 'BOOLEAN'
    return 'STRING'


class LagentTool(BaseAction):
    """A wrapper to align with the interface of Lagent tools."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(DefaultParser)  # Use string input & output
        self.tool = tool

        parameters = []
        required = []
        for p in tool.inputs:
            parameters.append(
                dict(
                    name=p.name,
                    description=p.description,
                    type=convert_type(p.type),
                ))
            if not p.optional:
                required.append(p.name)

        self._is_toolkit = False
        super().__init__(
            description=dict(
                name=tool.name,
                description=tool.toolmeta.description,
                parameters=parameters,
                required=required,
            ),
            enable=True,
        )

    def run(self, **kwargs) -> ActionReturn:

        try:
            outputs = self.tool(**kwargs)
            results = []

            if not isinstance(outputs, tuple):
                outputs = [outputs]

            for out, p in zip(outputs, self.tool.outputs):
                if p.type is ImageIO:
                    results.append(dict(type='image', content=out))
                elif p.type is AudioIO:
                    results.append(dict(type='audio', content=out))
                elif p.type is File:
                    results.append(dict(type='file', content=out))
                else:
                    results.append(dict(type='text', content=str(out)))

            return ActionReturn(
                type=self.name,
                args=kwargs,
                result=results,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=repr(e),
                args=kwargs,
                state=ActionStatusCode.API_ERROR,
            )
