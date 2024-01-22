import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from typing_extensions import Annotated, get_args, get_origin

from agentlego.parsers import DefaultParser
from agentlego.schema import Parameter, ToolMeta
from agentlego.types import CatgoryToIO


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
    def _collect_inputs(cls) -> Tuple[Parameter, ...]:
        inputs = []
        for p in inspect.signature(cls.apply).parameters.values():
            if p.name == 'self':
                continue

            annotation = p.annotation
            info = None
            if get_origin(annotation) is Annotated:
                for item in get_args(annotation):
                    if isinstance(item, Parameter):
                        info = item
                annotation = get_args(annotation)[0]

            input_ = Parameter(
                name=p.name,
                type=annotation,
                optional=p.default is not inspect._empty,
                default=p.default if p.default is not inspect._empty else None,
            )
            if info is not None:
                input_.update(info)
            inputs.append(input_)
        return tuple(inputs)

    @classmethod
    def _collect_outputs(cls) -> Optional[Tuple[Parameter, ...]]:
        outputs = []
        return_ann = inspect.signature(cls.apply).return_annotation
        if return_ann is inspect._empty:
            return None
        elif get_origin(return_ann) is tuple:
            annotations = get_args(return_ann)
            assert len(annotations) > 1 and Ellipsis not in annotations, (
                f'The number of outputs of `{cls.__name__}.apply` '
                'is undefined. Please specify like `Tuple[int, int, str]`')
        else:
            annotations = (return_ann, )

        for annotation in annotations:
            info = None
            if get_origin(annotation) is Annotated:
                for item in get_args(annotation):
                    if isinstance(item, Parameter):
                        info = item
                annotation = get_args(annotation)[0]

            output = Parameter(type=annotation)
            if info is not None:
                output.update(info)
            outputs.append(output)
        return tuple(outputs)

    @classmethod
    def get_default_toolmeta(cls, override=None) -> ToolMeta:
        toolmeta = override or getattr(cls, 'DEFAULT_TOOLMETA', {})
        toolmeta = copy.deepcopy(toolmeta)
        if isinstance(toolmeta, dict):
            toolmeta = ToolMeta(**toolmeta)

        if toolmeta.name is None:
            toolmeta.name = getattr(cls, 'default_name', cls.__name__)

        if toolmeta.description is None:
            doc = (cls.default_desc or '').strip()
            toolmeta.description = doc.partition('\n\n')[0].replace('\n', ' ')

        supported_types = set(CatgoryToIO.values())

        inputs = cls._collect_inputs()
        new_inputs = []
        if toolmeta.inputs is None:
            toolmeta.inputs = inputs
        else:
            assert len(inputs) == len(
                toolmeta.inputs), ('The length of `inputs` in toolmeta is different with '
                                   f'the number of arguments of `{cls.__name__}.apply`.')
        for i, item in enumerate(toolmeta.inputs):
            if isinstance(item, str):
                item = Parameter(type=CatgoryToIO[item])
            assert isinstance(item, Parameter), \
                ('The type of elements in inputs should be `str` '
                 f'or `Parameter`, got `{type(item)}` instead.')
            inputs[i].update(item)
            new_inputs.append(inputs[i])
            assert inputs[i].type is not inspect._empty, (
                f'The type of input `{inputs[i].name}` of '
                f'`{cls.__name__}` is not specified.')
            assert inputs[i].type in supported_types, (f'The type of input `{inputs[i].name}` of '
                                                       f'`{cls.__name__}` is not supported. '
                                                       f'Supported types are {supported_types}')
        toolmeta.inputs = tuple(new_inputs)

        outputs = cls._collect_outputs()
        new_outputs = []
        if toolmeta.outputs is None:
            assert outputs is not None, (
                f'The type of output of `{cls.__name__}` is not specified.')
            toolmeta.outputs = outputs
        elif toolmeta.outputs is not None and outputs is not None:
            assert len(outputs) == len(
                toolmeta.outputs), ('The length of `outputs` in toolmeta is different with '
                                    f'the type hint of return value of `{cls.__name__}.apply`.')
        for i, item in enumerate(toolmeta.outputs):
            if isinstance(item, str):
                item = Parameter(type=CatgoryToIO[item])
            assert isinstance(item, Parameter), \
                ('The type of elements in outputs should be `str` '
                 f'or `Parameter`, got `{type(item)}` instead.')
            if outputs is not None:
                outputs[i].update(item)
                item = outputs[i]
            new_outputs.append(item)
            assert item.type in supported_types, (
                f'The type of output of `{cls.__name__}` is not supported. '
                f'Supported types are {supported_types}')
        toolmeta.outputs = tuple(new_outputs)

        return toolmeta

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
