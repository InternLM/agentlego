# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any, Dict, Tuple

from .type_mapping_parser import TypeMappingParser


class VisualChatGPTParser(TypeMappingParser):
    _default_agent_cat2type = {
        'image': 'path',
        'text': 'string',
    }
    _file_suffix = {
        'image': 'png',
    }

    def parse_inputs(self, *args, **kwargs) -> Tuple[Tuple, Dict]:

        # split single string into multiple inputs
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple(s.strip() for s in args[0].split(','))
            kwargs = {}

        return super().parse_inputs(*args, **kwargs)

    def parse_outputs(self, outputs: Any) -> str:
        outputs = super().parse_outputs(outputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        # gather outputs into a single string
        return ', '.join(str(output) for output in outputs)

    def refine_description(self, description: str) -> str:
        refined = super().refine_description(description)
        num_inputs = len(self.description_to_inputs(description))

        if num_inputs > 1:
            refined += ' Inputs should be separated by comma.'

        return refined


class HuggingFaceAgentParser(TypeMappingParser):
    _default_agent_cat2type = {
        'image': 'pillow',
        'text': 'string',
    }

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            return match.group(2).strip()

        return re.sub(r'{{{(input|output):\s*(.*?)}}}', _reformat, description)
