import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import Parameter, ToolMeta
from .utils.parameters import extract_toolmeta


class BaseTool(metaclass=ABCMeta):
    default_desc: Optional[str] = None

    def __init__(
        self,
        toolmeta: Union[dict, ToolMeta, None] = None,
        parser: Callable = DefaultParser,
    ):
        self.toolmeta = self.get_default_toolmeta(toolmeta)
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

    @property
    def inputs(self) -> Tuple[Parameter, ...]:
        return self.toolmeta.inputs

    @property
    def arguments(self) -> Mapping[str, Parameter]:
        return {i.name: i for i in self.toolmeta.inputs}

    @property
    def outputs(self) -> Tuple[Parameter, ...]:
        return self.toolmeta.outputs

    @classmethod
    def get_default_toolmeta(cls, override=None) -> ToolMeta:
        if isinstance(override, dict):
            override = ToolMeta(**override)
        override = ToolMeta() if override is None else copy.deepcopy(override)

        if override.name is None:
            override.name = cls.__name__

        if override.description is None:
            doc = (cls.default_desc or '').partition('\n\n')[0].replace('\n', ' ')
            override.description = doc.strip()

        return extract_toolmeta(cls.apply, override=override)

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
