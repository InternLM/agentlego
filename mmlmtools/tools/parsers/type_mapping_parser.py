# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any, Callable, Optional

from PIL import Image

from mmlmtools.utils import get_new_image_name
from .base_parser import BaseParser


class TypeMappingParser(BaseParser):
    TypeMapping: dict[str, str] = {}

    def __init__(self, type_mapping: Optional[dict[str, str]] = None):

        if type_mapping is not None:
            self.type_mapping = type_mapping.copy()
        else:
            self.type_mapping = self.TypeMapping.copy()

    def description_to_input_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{input:[ ]*(.*?)[ ]*}}}', description))

    def description_to_output_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{output:[ ]*(.*?)[ ]*}}}', description))

    def _input_path_to_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _input_pil_to_image(self, image: Image.Image) -> Image.Image:
        return image.convert('RGB')

    def _input_str_to_text(self, text: str) -> str:
        return text

    def _output_image_to_path(self, image: Image.Image) -> str:
        path = get_new_image_name('image/temp.jpg', func_name='temp')
        image.save(path)
        return path

    def _output_image_to_pil(self, image: Image.Image) -> Image.Image:
        return image

    def _output_text_to_str(self, text: str) -> str:
        return text

    def _get_input_formatter(self, data_type: str,
                             data_format: str) -> Callable:

        if data_type == 'image':
            if data_format == 'path':
                return self._input_path_to_image
            elif data_format == 'PIL Image':
                return self._input_pil_to_image
        elif data_type == 'text':
            if data_format == 'str':
                return self._input_str_to_text

        raise NotImplementedError

    def _get_output_formatter(self, data_type: str,
                              data_format: str) -> Callable:

        if data_type == 'image':
            if data_format == 'path':
                return self._output_image_to_path
            elif data_format == 'PIL Image':
                return self._output_image_to_pil
        elif data_type == 'text':
            if data_format == 'str':
                return self._output_text_to_str

        raise NotImplementedError

    def _format_input(self, input: Any, data_type: str) -> Any:
        if data_type not in self.type_mapping:
            raise ValueError
        data_format = self.type_mapping[data_type]
        formatter = self._get_input_formatter(data_type, data_format)
        return formatter(input)

    def _format_output(self, output: Any, data_type: str) -> Any:
        if data_type not in self.type_mapping:
            raise ValueError
        data_format = self.type_mapping[data_type]
        formatter = self._get_output_formatter(data_type, data_format)
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
            if data_type not in self.type_mapping:
                raise ValueError
            data_format = self.type_mapping[data_type]

            return f'{data_type} represented in {data_format}'

        num_inputs = len(self.description_to_input_types(description))

        description = re.sub(r'{{{(input|output):[ ]*(.*?)}}}', _reformat,
                             description)

        if num_inputs > 1:
            description += ' Inputs should be separated by comma.'

        return description
