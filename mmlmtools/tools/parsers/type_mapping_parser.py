# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any, Callable, Optional

from PIL import Image

from mmlmtools.utils import get_new_image_name
from .base_parser import BaseParser


class formatter:

    def __init__(self,
                 input_type: Optional[str] = None,
                 output_type: Optional[str] = None,
                 format: Optional[str] = None):

        if input_type and output_type:
            raise ValueError(
                '`input_type` and `output_type` cannot be set at the same time'
            )
        if not (input_type or output_type):
            raise ValueError(
                'Either `input_type` or `output_type` should be set.')
        if not format:
            raise ValueError('format cannot be None')

        self.type = input_type or output_type
        self.format = format
        self.dict_name = ('_input_formatters'
                          if input_type else '_output_formatters')

    def __call__(self, func: Callable):
        self.func = func
        return self

    def __get__(self, obj, objtype=None) -> Callable:
        if obj is None:
            return self.func
        return lambda *args, **kwargs: self.func(obj, *args, **kwargs)

    def __set_name__(self, owner: Any, name: str) -> None:
        if not hasattr(owner, self.dict_name):
            setattr(owner, self.dict_name, {})
        getattr(owner, self.dict_name)[(self.type, self.format)] = name


class TypeMappingParser(BaseParser):
    _default_type_mapping: dict[str, str] = {}
    _type_mapping: dict[str, str]
    _input_formatters: dict[tuple[str, str], str]
    _output_formatters: dict[tuple[str, str], str]

    def __init__(self, type_mapping: Optional[dict[str, str]] = None):

        if type_mapping is not None:
            self._type_mapping = type_mapping.copy()
        else:
            self._type_mapping = self._default_type_mapping.copy()

    @formatter(input_type='image', format='path')
    def _path_to_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    @formatter(output_type='image', format='path')
    def _image_to_path(self, image: Image.Image) -> str:
        path = get_new_image_name('image/temp.jpg', func_name='temp')
        image.save(path)
        return path

    @formatter(input_type='image', format='PIL Image')
    def _pil_to_image(self, image: Image.Image) -> Image.Image:
        return image.convert('RGB')

    @formatter(output_type='image', format='PIL Image')
    def _image_to_pil(self, image: Image.Image) -> Image.Image:
        return image.convert('RGB')

    @formatter(input_type='text', format='str')
    def _str_to_text(self, text: str) -> str:
        return text

    @formatter(output_type='text', format='str')
    def _text_to_str(self, text: str) -> str:
        return text

    def _format_input(self, input: Any, type: str) -> Any:
        if type not in self._type_mapping:
            raise ValueError
        format = self._type_mapping[type]
        formatter = getattr(self, self._input_formatters[(type, format)], None)
        if formatter is None:
            raise ValueError(
                f'No formatter for input type `{type}` and format `{format}`')
        return formatter(input)

    def _format_output(self, output: Any, type: str) -> Any:
        if type not in self._type_mapping:
            raise ValueError
        format = self._type_mapping[type]
        formatter = getattr(self, self._output_formatters[(type, format)],
                            None)
        if formatter is None:
            raise ValueError(
                f'No formatter for output type `{type}` and format `{format}`')
        return formatter(output)

    def _split_inputs(self, inputs: Any) -> tuple:
        return tuple(s.strip() for s in inputs.split(','))

    def _gather_outputs(self, outputs: tuple) -> Any:
        return ', '.join(outputs)

    def parse_inputs(self, inputs: tuple, input_types: tuple[str]) -> tuple:

        if len(inputs) == 1 and isinstance(inputs[0], str):
            inputs = self._split_inputs(inputs[0])

        if len(inputs) != len(input_types):
            raise ValueError

        return tuple(
            self._format_input(input, data_type)
            for input, data_type in zip(inputs, input_types))

    def parse_outputs(self, outputs: Any, output_types: tuple[str]) -> Any:
        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        if len(outputs) != len(output_types):
            raise ValueError

        outputs = tuple(
            self._format_output(output, data_type)
            for output, data_type in zip(outputs, output_types))

        return self._gather_outputs(outputs)

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            data_type = match.group(2).strip()
            if data_type not in self._type_mapping:
                raise ValueError
            data_format = self._type_mapping[data_type]

            return f'{data_type} represented in {data_format}'

        num_inputs = len(self.description_to_input_types(description))

        description = re.sub(r'{{{(input|output):[ ]*(.*?)}}}', _reformat,
                             description)

        if num_inputs > 1:
            description += ' Inputs should be separated by comma.'

        return description

    def description_to_input_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{input:[ ]*(.*?)[ ]*}}}', description))

    def description_to_output_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{output:[ ]*(.*?)[ ]*}}}', description))
