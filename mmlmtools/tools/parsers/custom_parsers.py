# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any

from .type_mapping_parser import TypeMappingParser


class VisualChatGPTParser(TypeMappingParser):
    _default_agent_type2format = {
        'image': 'path',
        'text': 'str',
    }

    def parse_inputs(self, inputs: tuple) -> tuple:

        # split single string into multiple inputs
        if len(inputs) == 1 and isinstance(inputs[0], str):
            inputs = tuple(s.strip() for s in inputs[0].split(','))

        return super().parse_inputs(inputs)

    def parse_outputs(self, outputs: Any) -> str:
        outputs = super().parse_outputs(outputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        # gather outputs into a single string
        return ', '.join(str(output) for output in outputs)

    def refine_description(self, description: str) -> str:
        refined = super().refine_description(description)
        num_inputs = len(self.description_to_input_types(description))

        if num_inputs > 1:
            refined += ' Inputs should be separated by comma.'

        return refined


class HuggingFaceAgentParser(TypeMappingParser):
    _default_agent_type2format: dict[str, str] = {
        'image': 'pillow',
        'text': 'str',
    }

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            return match.group(2).strip()

        return re.sub(r'{{{(input|output):[ ]*(.*?)}}}', _reformat,
                      description)
