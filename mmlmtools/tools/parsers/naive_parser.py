# Copyright (c) OpenMMLab. All rights reserved.
import re

from .base_parser import BaseParser


class NaiveParser(BaseParser):

    def parse_inputs(self, inputs: str | tuple) -> tuple:
        if isinstance(inputs, str):
            return inputs,
        return inputs

    def parse_outputs(self, outputs):
        return outputs

    def refine_description(self, description: str) -> str:
        f'''Refine the tool description by replacing the input and output
        markers with raw data types. For example, ```"{{{input: image}}}"```
        will be replaced with ```"image"```.

        Args:
            description (str): The original tool description.

        Returns:
            str: The refined tool description.
        '''

        def _remove_brackets(matched: re.Match) -> str:
            return matched.group(2).strip()

        return re.sub(r'{{{(input|output): (.*?)}}}', _remove_brackets,
                      description)

    def description_to_input_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{input:[ ]*(.*?)[ ]*}}}', description))

    def description_to_output_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{output:[ ]*(.*?)[ ]*}}}', description))
