# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect

from langchain.tools import StructuredTool

from agentlego.parsers import DefaultParser
from ..base import BaseTool


def construct_langchain_tool(tool: BaseTool):
    tool = copy.copy(tool)
    tool.set_parser(DefaultParser)  # Use string input & output

    def call(*args, **kwargs):
        return tool(*args, **kwargs)

    call_args = {}
    call_params = []
    for arg_name in tool.input_fields:
        call_args[arg_name] = str
        call_params.append(
            inspect.Parameter(
                arg_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str))
    call.__signature__ = inspect.Signature(call_params)
    call.__annotations__ = call_args

    return StructuredTool.from_function(
        func=call,
        name=tool.name,
        description=tool.toolmeta.description,
    )
