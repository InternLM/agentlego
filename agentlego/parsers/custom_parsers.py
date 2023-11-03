# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple

from .default_parser import DefaultParser


class LangChainParser(DefaultParser):
    agent_cat2type = {
        'image': 'path',
        'text': 'string',
        'audio': 'path',
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
        num_inputs = len(self.toolmeta.inputs)

        if num_inputs > 1:
            refined += ' Inputs should be separated by comma.'

        return refined


class HuggingFaceAgentParser(DefaultParser):
    agent_cat2type = {}

    def parse_outputs(self, outputs):
        from transformers.tools.agent_types import (AgentAudio, AgentImage,
                                                    AgentText, AgentType)

        if isinstance(outputs, tuple):
            assert len(outputs) == len(self.toolmeta.outputs)
        else:
            assert len(self.toolmeta.outputs) == 1
            outputs = [outputs]

        parsed_outs = []
        for tool_output, out_category in zip(outputs, self.toolmeta.outputs):
            if out_category == 'image':
                out = AgentImage(tool_output.to_pil())
            elif out_category == 'text':
                out = AgentText(tool_output)
            elif out_category == 'audio':
                out = AgentAudio(
                    tool_output.to_tensor().flatten(),
                    samplerate=tool_output.sampling_rate)
            else:
                out = AgentType(tool_output)
            parsed_outs.append(out)

        return parsed_outs[0] if len(parsed_outs) == 1 else parsed_outs

    def refine_description(self) -> str:
        inputs_desc = []
        for in_category in self.toolmeta.inputs:
            inputs_desc.append(f'{in_category}')
        inputs_desc = 'Args: ' + ', '.join(inputs_desc)

        description = f'{self.toolmeta.description} {inputs_desc}'

        return description
