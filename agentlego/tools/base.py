# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from abc import ABCMeta, abstractmethod
from types import MethodType
from typing import Any, Callable, Dict, Union

from agentlego.schema import Parameter, ToolMeta


class BaseTool(metaclass=ABCMeta):

    def __init__(self, toolmeta: Union[dict, ToolMeta], parser: Callable):
        toolmeta = copy.deepcopy(toolmeta)
        if isinstance(toolmeta, dict):
            toolmeta = ToolMeta(**toolmeta)
        self.toolmeta = toolmeta
        self.set_parser(parser)
        self._is_setup = False

    @property
    def name(self) -> str:
        return self.toolmeta.name

    @name.setter
    def name(self, val: str):
        self.toolmeta.name = val

    @property
    def description(self) -> str:
        return self.parser.refine_description()

    @description.setter
    def description(self, val: str):
        self.toolmeta.description = val

    def set_parser(self, parser: Callable):
        self.parser = parser(self)
        self._parser_constructor = parser

    def setup(self):
        """Implement lazy initialization here that will be performed before the
        first call of ```apply()```, for example loading the model."""
        self._is_setup = True

    def __call__(self, *args: Any, **kwargs) -> Any:

        if not self._is_setup:
            self.setup()
            self._is_setup = True

        inputs, kwinputs = self.parser.parse_inputs(*args, **kwargs)

        outputs = self.apply(*inputs, **kwinputs)

        results = self.parser.parse_outputs(outputs)
        return results

    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def __repr__(self) -> str:
        repr_str = (f'{type(self).__name__}('
                    f'toolmeta={self.toolmeta}, '
                    f'parser={type(self.parser).__name__})')
        return repr_str

    @property
    def parameters(self) -> Dict[str, Parameter]:
        parameters = {}
        for category, p in zip(
                self.toolmeta.inputs,
                inspect.signature(self.apply).parameters.values()):
            if isinstance(self.apply, MethodType) and p.name == 'self':
                continue
            parameters[p.name] = Parameter(
                name=p.name,
                category=category,
                description=None,
                optional=p.default != inspect._empty,
                default=p.default if p.default != inspect._empty else None,
            )
        return parameters

    def __copy__(self):
        obj = object.__new__(type(self))
        obj.__dict__.update(self.__dict__)
        obj.toolmeta = copy.deepcopy(self.toolmeta)
        obj.set_parser(self._parser_constructor)
        return obj

    def to_transformers_agent(self):
        from .wrappers.transformers_agent import TransformersAgentTool
        return TransformersAgentTool(self)

    def to_langchain(self):
        from .wrappers.langchain import construct_langchain_tool
        return construct_langchain_tool(self)

    def to_lagent(self):
        from .wrappers.lagent import LagentTool
        return LagentTool(self)
