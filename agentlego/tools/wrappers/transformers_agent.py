# Copyright (c) OpenMMLab. All rights reserved.
import copy

from transformers.tools import Tool
from transformers.tools.agent_types import (AgentAudio, AgentImage, AgentText,
                                            AgentType)

from agentlego.parsers import NaiveParser
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, ImageIO


def cast_lego_to_hf(value):
    if isinstance(value, ImageIO):
        return AgentImage(value.to_pil())
    elif isinstance(value, AudioIO):
        return AgentAudio(value.to_tensor().flatten(), value.sampling_rate)
    elif isinstance(value, str):
        return AgentText(value)
    else:
        return AgentType(value)


class TransformersAgentTool(Tool):
    """Adapter for agentlego.tools.Tool to transformers.tools.Tool."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(NaiveParser)  # Use raw input & output
        self.tool = tool

        # remove spaces in the tool name which is not allowed in the
        # huggingface agent system
        self.name: str = 'agentlego_' + tool.name.lower().replace(' ', '_')

        inputs_desc = []
        for arg_name, in_category in zip(tool.input_fields,
                                         tool.toolmeta.inputs):
            inputs_desc.append(f'{arg_name} ({in_category})')
        inputs_desc = 'Args: ' + ', '.join(inputs_desc)
        self.description: str = f'{tool.toolmeta.description} {inputs_desc}'

        self.inputs = list(tool.toolmeta.inputs)
        self.outputs = list(tool.toolmeta.outputs)

    def __call__(self, *args, **kwargs):
        for k, v in zip(self.tool.input_fields, args):
            kwargs[k] = v

        parsed_kwargs = {}
        for k, v in kwargs.items():
            in_category = self.inputs[self.tool.input_fields.index(k)]
            if in_category == 'audio':
                parsed_kwargs[k] = AudioIO(v)
            elif in_category == 'image':
                parsed_kwargs[k] = ImageIO(v)
            elif in_category == 'text':
                parsed_kwargs[k] = str(v)

        outputs = self.tool(**parsed_kwargs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        parsed_outs = [cast_lego_to_hf(out) for out in outputs]

        return parsed_outs[0] if len(parsed_outs) == 1 else parsed_outs
