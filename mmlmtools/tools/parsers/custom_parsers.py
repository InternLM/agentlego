# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any

from .type_mapping_parser import TypeMappingParser


class VisualChatGPTParser(TypeMappingParser):
    TypeMapping: dict[str, str] = {
        'image': 'path',
        'text': 'str',
    }


class HuggingFaceAgentParser(TypeMappingParser):
    TypeMapping: dict[str, str] = {
        'image': 'PIL Image',
        'text': 'str',
    }

    def _split_inputs(self, inputs: Any) -> tuple:
        return inputs if isinstance(inputs, tuple) else (inputs, )

    def _gather_outputs(self, outputs: tuple) -> Any:
        return outputs[0] if len(outputs) == 1 else outputs

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            return match.group(2).strip()

        return re.sub(r'{{{(input|output):[ ]*(.*?)}}}', _reformat,
                      description)
