# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Tuple

from .base_parser import BaseParser


class NaiveParser(BaseParser):

    def parse_inputs(self, *args, **kwargs) -> Tuple[tuple, dict]:
        return args, kwargs

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

        def _reformat(match: re.Match) -> str:
            return match.group(2).strip()

        return re.sub(r'{{{(input|output): (.*?)}}}', _reformat, description)
