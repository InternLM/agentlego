import copy
import inspect
from typing import Callable, Optional, Tuple, Union

from typing_extensions import Annotated, get_args, get_origin

from agentlego.schema import Parameter, ToolMeta
from agentlego.types import CatgoryToIO


def get_input_parameters(func: Callable) -> Tuple[Parameter, ...]:
    inputs = []
    for p in inspect.signature(func).parameters.values():
        if p.name == 'self':
            continue

        annotation = p.annotation
        info = None
        if get_origin(annotation) is Annotated:
            for item in get_args(annotation):
                if isinstance(item, Parameter):
                    info = item
            annotation = get_args(annotation)[0]
        if get_origin(annotation) is Union:
            types = [i for i in get_args(annotation) if i is not type(None)]
            assert len(types) == 1, (f'The union type of input `{p.name}` in '
                                     f'`{func.__qualname__}` is not supported.')
            annotation = types[0]

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


def get_output_parameters(func: Callable) -> Optional[Tuple[Parameter, ...]]:
    outputs = []
    return_ann = inspect.signature(func).return_annotation
    if return_ann is inspect._empty:
        return None
    elif get_origin(return_ann) is tuple:
        annotations = get_args(return_ann)
        assert len(annotations) > 1 and Ellipsis not in annotations, (
            f'The number of outputs of `{func.__qualname__}` '
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


def extract_toolmeta(func: Callable, override: Optional[ToolMeta] = None) -> ToolMeta:
    supported_types = set(CatgoryToIO.values())

    inputs = get_input_parameters(func)
    if override is not None and override.inputs is not None:
        assert len(inputs) == len(
            override.inputs), ('The length of `inputs` in toolmeta is different with '
                               f'the number of arguments of `{func.__qualname__}`.')
        for input_, new_input in zip(inputs, override.inputs):
            input_.update(new_input)
    for input_ in inputs:
        assert input_.type is not inspect._empty, (
            f'The type of input `{input_.name}` of '
            f'`{func.__qualname__}` is not specified.')
        assert input_.type in supported_types, (
            f'The type of input `{input_.name}` of {func.__qualname__}` '
            'is not supported. Supported types are ' +
            ', '.join(i.__name__ for i in supported_types))

    outputs = get_output_parameters(func)
    if outputs is None:
        assert override is not None and override.outputs is not None, (
            f'The type of output of `{func.__qualname__}` is not specified.')
        outputs = override.outputs
    elif override is not None and override.outputs is not None:
        assert len(outputs) == len(override.outputs), (
            'The length of `outputs` in toolmeta is different with '
            f'the type hint of return value of `{func.__qualname__}`.')
        for output, new_output in zip(outputs, override.outputs):
            output.update(new_output)
    for output in outputs:
        assert output.type is not inspect._empty, (
            f'The type of output `{output.name}` of '
            f'`{func.__qualname__}` is not specified.')
        assert output.type in supported_types, (
            f'The type of return value of {func.__qualname__}` '
            'is not supported. Supported types are ' +
            ', '.join(i.__name__ for i in supported_types))

    if override:
        toolmeta = copy.deepcopy(override)
        toolmeta.inputs = tuple(inputs)
        toolmeta.outputs = tuple(outputs)
    else:
        toolmeta = ToolMeta(inputs=inputs, outputs=outputs)

    return toolmeta
