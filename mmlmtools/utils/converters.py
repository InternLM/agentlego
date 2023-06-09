# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Union

from .utils import get_new_image_name

DataType = Union[str, tuple[str]]


def add_converter(data_type: str, data_style: str):
    ...


class DataConverter():
    supported_data_type = ['image']

    def __init__(self, type_to_style: dict[str, str]):
        self.type_to_style = type_to_style

    def convert_inputs(self, inputs, input_types: DataType) -> Any:
        results = []
        for input_type in input_types:
            ...

        return results

    def generate_input_description(self, input_types: DataType) -> str:
        return ''

    @add_converter('image', 'image_path')
    def _image_from_path(self, inputs):
        return inputs

    @add_converter('image', 'pil')
    def _image_from_pil(self, inputs):
        temp_image_path = get_new_image_name(
            'image/temp.jpg', func_name='temp')
        inputs.save(temp_image_path)
        return temp_image_path


class VisualChatGPTConverter(DataConverter):
    supported_data_type = ['image', 'text']

    def __init__(self):
        self.type_to_style = {
            'image': 'image_path',
            'text': 'text',
        }
