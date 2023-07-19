# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import re
from types import FunctionType
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image

from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import get_new_image_name
from .base_parser import BaseParser


class formatter:

    __get__ = FunctionType.__get__

    def __init__(self, type: str, source: str, target: str):
        self.type = type
        self.source = source
        self.target = target

    def __call__(self, func: Callable):
        self.func = func
        return self

    def __set_name__(self, owner: Any, name: str) -> None:
        if not hasattr(owner, '_formatters'):
            setattr(owner, '_formatters', {})

        # `formatters` is a dict of data type and format to function name with
        # the following structure: dict[(type, source, target)] = name
        formatters = getattr(owner, '_formatters')

        key = (self.type, self.source, self.target)
        if key in formatters:
            raise ValueError(
                f'Duplicated formatters for type `{self.type}`, source '
                f'`{self.source}` and target `{self.target}` already exists.')

        formatters[key] = name


class TypeMappingParser(BaseParser):
    # map data type (e.g. image) to format (e.g. path) on the agent side
    _agent_type2format: dict[str, str]
    # default type mapping that will be used if no type mapping is provided
    _default_agent_type2format: dict[str, str] = {}
    # (type, source, target) -> name
    _formatters: dict[tuple[str, str, str], str]

    _allowed_toolarg2format: dict[str, dict[type, str]] = {
        'image': {
            str: 'path',
            Image.Image: 'pillow',
            np.ndarray: 'ndarray'
        },
        'text': {
            str: 'str',
        }
    }

    def __init__(self, type2format: Optional[dict[str, str]] = None):

        if type2format is not None:
            self._agent_type2format = type2format.copy()
        else:
            self._agent_type2format = self._default_agent_type2format.copy()

        # The input/output formatters will be determined when bound to a tool
        self._input_formatters: Optional[list[Callable]] = None
        self._output_formatters: Optional[list[Callable]] = None

    def bind_tool(self, tool: Any) -> None:
        assert hasattr(tool, 'apply') and callable(tool.apply)
        assert hasattr(tool, 'toolmeta') and isinstance(
            tool.toolmeta, ToolMeta)

        input_types = tool.input_types
        output_types = tool.output_types

        agent_input_formats = [self._agent_type2format[t] for t in input_types]
        agent_output_formats = [
            self._agent_type2format[t] for t in output_types
        ]
        tool_input_formats = []
        tool_output_formats = []

        # parser tool input formats
        tool_argspec = inspect.getfullargspec(tool.apply)
        if len(tool_argspec.args) != len(input_types) + 1:
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` should have '
                f'{len(input_types)} argument(s) (excluding `self`) indicated'
                f' by the description, but got {len(tool_argspec.args) - 1} ')

        for t, arg in zip(input_types, tool_argspec.args[1:]):
            argtype = tool_argspec.annotations.get(arg, None)
            if argtype is None:
                raise ValueError(
                    f'Argument `{arg}` of the `apply` method of the tool '
                    f'`{tool.name}` should have type annotation.')

            argformat = self._allowed_toolarg2format[t].get(argtype, None)
            if argformat is None:
                raise ValueError(
                    f'Argument `{arg}` of the `apply` method of the tool '
                    f'`{tool.name}` havs type annotation `{argtype}`, '
                    f'which is not supported for data type {t}.')

            tool_input_formats.append(argformat)

        # parse tool output formats
        if 'return' not in tool_argspec.annotations:
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` should have '
                f'return type annotation.')

        returns = tool_argspec.annotations['return']
        if not isinstance(returns, tuple):
            returns = (returns, )

        if len(returns) != len(output_types):
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` '
                f'should have {len(output_types)} return(s) indicated '
                f'by the description, but got {len(returns)}.')

        for i, (t, rettype) in enumerate(zip(output_types, returns)):
            retformat = self._allowed_toolarg2format[t].get(rettype, None)
            if retformat is None:
                raise ValueError(f'The {i}-th return of the `apply` method of '
                                 f'the tool `{tool.name}` has type '
                                 f'annotation `{rettype}`, which is not '
                                 f'supported for data type {t}.')

            tool_output_formats.append(retformat)

        self._input_formatters = []
        for t, source, target in zip(input_types, agent_input_formats,
                                     tool_input_formats):
            if source == target:
                self._input_formatters.append(lambda x: x)
            else:
                if (t, source, target) not in self._formatters:
                    raise ValueError(
                        f'No formatter for input type `{t}`, source '
                        f'`{source}` and target `{target}`, required by tool '
                        f'`{tool.name}`.')
                self._input_formatters.append(
                    getattr(self, self._formatters[(t, source, target)]))

        self._output_formatters = []
        for t, source, target in zip(input_types, tool_output_formats,
                                     agent_output_formats):
            if source == target:
                self._output_formatters.append(lambda x: x)
            else:
                if (t, source, target) not in self._formatters:
                    raise ValueError(
                        f'No formatter for output type `{t}`, source '
                        f'`{source}`  and target `{target}`, required by tool '
                        f'`{tool.name}`.')
                self._output_formatters.append(
                    getattr(self, self._formatters[(t, source, target)]))

    def _get_formatter(self, type: str, source: str, target: str) -> Callable:
        if source == target:
            return lambda x: x
        return getattr(self, self._formatters[(type, source, target)])

    def parse_inputs(self, inputs: Any) -> tuple:
        if self._input_formatters is None:
            raise RuntimeError('The parser is not bound to a tool yet.')

        if not isinstance(inputs, tuple):
            inputs = (inputs, )

        if len(inputs) != len(self._input_formatters):
            raise ValueError(
                f'Failed to parse {len(self._input_formatters)} inputs')

        return tuple(
            formatter(input)
            for input, formatter in zip(inputs, self._input_formatters))

    def parse_outputs(self, outputs: Any) -> Any:
        if self._output_formatters is None:
            raise RuntimeError('The parser is not bound to a tool yet.')

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        if len(outputs) != len(self._output_formatters):
            raise ValueError(f'Expect {len(self._output_formatters)} outputs, '
                             f'but got {len(outputs)}.')

        outputs = tuple(
            formatter(output)
            for output, formatter in zip(outputs, self._output_formatters))

        return outputs[0] if len(outputs) == 1 else outputs

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            data_type = match.group(2).strip()
            if data_type not in self._agent_type2format:
                raise ValueError
            data_format = self._agent_type2format[data_type]

            return f'{data_type} represented in {data_format}'

        return re.sub(r'{{{(input|output):[ ]*(.*?)}}}', _reformat,
                      description)

    def description_to_input_types(self, description: str) -> tuple[str]:
        input_types = tuple(
            re.findall(r'{{{input:[ ]*(.*?)[ ]*}}}', description))
        for t in input_types:
            if t not in self._allowed_toolarg2format:
                raise ValueError(f'Unknown input type `{t}`')
        return input_types

    def description_to_output_types(self, description: str) -> tuple[str]:
        output_types = tuple(
            re.findall(r'{{{output:[ ]*(.*?)[ ]*}}}', description))
        for t in output_types:
            if t not in self._allowed_toolarg2format:
                raise ValueError(f'Unknown input type `{t}`')
        return output_types

    @formatter(type='image', source='path', target='pillow')
    def _image_path_to_pil(self, path: str) -> Image.Image:
        return Image.open(path)

    @formatter(type='image', source='pillow', target='path')
    def _image_pil_to_path(self, image: Image.Image) -> str:
        path = get_new_image_name('image/temp.jpg', func_name='temp')
        image.save(path)
        return path

    @formatter(type='image', source='pillow', target='ndarray')
    def _image_pil_to_ndarray(self, image: Image.Image) -> np.ndarray:
        return np.array(image)

    @formatter(type='image', source='ndarray', target='pillow')
    def _image_ndarray_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    @formatter(type='image', source='ndarray', target='path')
    def _image_ndarray_to_path(self, image: np.ndarray) -> str:
        path = get_new_image_name('image/temp.jpg', func_name='temp')
        cv2.imwrite(path, image)
        return path

    @formatter(type='image', source='path', target='ndarray')
    def _image_path_to_ndarray(self, path: str) -> np.ndarray:
        return cv2.imread(path)
