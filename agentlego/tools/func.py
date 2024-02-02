from copy import deepcopy
from inspect import cleandoc
from typing import Callable, Optional, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from .base import BaseTool
from .utils.parameters import extract_toolmeta


class _FuncTool(BaseTool):

    def __init__(self,
                 func: Callable,
                 toolmeta: ToolMeta,
                 parser: Callable = DefaultParser):
        self.func = func
        self.toolmeta = deepcopy(toolmeta)
        self.set_parser(parser)
        self._is_setup = True

    def apply(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _FuncToolType:

    def __init__(self, func: Callable, toolmeta: ToolMeta):
        self.func = func
        self.toolmeta = toolmeta

    def __call__(self,
                 toolmeta: Union[dict, ToolMeta, None] = None,
                 parser: Callable = DefaultParser):
        return _FuncTool(self.func, self.get_default_toolmeta(toolmeta), parser=parser)

    def get_default_toolmeta(self, override=None) -> ToolMeta:
        if override is None:
            return self.toolmeta

        override = deepcopy(override)
        override = ToolMeta(**override) if isinstance(override, dict) else override

        if override.name is None:
            override.name = self.toolmeta.name
        if override.description is None:
            override.description = self.toolmeta.description
        if override.inputs is None:
            override.inputs = self.toolmeta.inputs
        if override.outputs is None:
            override.outputs = self.toolmeta.outputs

        return override


def make_tool(func: Optional[Callable] = None,
              toolmeta: Optional[ToolMeta] = None,
              infer_meta: bool = True) -> Union[BaseTool, Callable]:
    """Make tool from function.

    Args:
        func (Callable | None): The execution function. If not specified, return a
            function decorator. Defaults to None.
        toolmeta (ToolMeta | dict | None): The meta information of the tool.
            Defaults to None.
        infer_meta (bool): Whether to infer the tool meta information. If False, directly
            use the input ``toolmeta``. If True, try to extract meta information and
            merge to the input toolmeta: Use function name as tool name; Use function
            docstring as description; Use type hint to infer inputs and outputs
            annotations. Defaults to True.

    Examples:
        .. code-block:: python
            from agentlego.tools import make_tool

            @make_tool
            def multiply(a: int, b: int) -> int:
                '''Multiply the input integers.'''
                return a * b

            @make_tool(toolmeta=dict(name="GetTime", description='Return the current time.'))
            def clock() -> str:
                from datetime import datetime
                return datetime.now().strftime('%Y/%m/%d %H:%M')
    """  # noqa: E501
    if isinstance(toolmeta, dict):
        toolmeta = ToolMeta(**toolmeta)

    def make_tool(func, override):
        if infer_meta:
            toolmeta = extract_toolmeta(func, override)
            if toolmeta.name is None:
                toolmeta.name = func.__name__
            if toolmeta.description is None and func.__doc__:
                toolmeta.description = cleandoc(func.__doc__).partition('\n\n')[0]
        else:
            toolmeta = deepcopy(override)
        tool = _FuncToolType(func, toolmeta=toolmeta)
        return tool

    if func is None:

        def wrapper(func: Callable):
            return make_tool(func, toolmeta)

        return wrapper
    else:
        return make_tool(func, override=toolmeta)
