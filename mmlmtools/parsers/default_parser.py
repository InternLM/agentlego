# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmlmtools.types import CatgoryToIO, IOType
from .base_parser import BaseParser


class DefaultParser(BaseParser):
    agent_cat2type = {
        'image': 'path',
        'text': 'string',
        'audio': 'path',
    }

    def parse_inputs(self, *args, **kwargs) -> Tuple[tuple, dict]:
        args = args + tuple(kwargs.values())
        assert len(args) == len(self.toolmeta.inputs)

        parsed_args = []
        for agent_input, in_category in zip(args, self.toolmeta.inputs):
            tool_type = CatgoryToIO[in_category]
            if not isinstance(agent_input, tool_type):
                tool_input = tool_type(agent_input)
            else:
                tool_input = agent_input
            parsed_args.append(tool_input)

        return parsed_args, {}

    def parse_outputs(self, outputs):
        if isinstance(outputs, tuple):
            assert len(outputs) == len(self.toolmeta.outputs)
        else:
            assert len(self.toolmeta.outputs) == 1
            outputs = [outputs]

        parsed_outs = []
        for tool_output, out_category in zip(outputs, self.toolmeta.outputs):
            agent_type = self.agent_cat2type[out_category]
            if isinstance(tool_output, IOType):
                tool_output = tool_output.to(agent_type)
            parsed_outs.append(tool_output)

        return parsed_outs[0] if len(parsed_outs) == 1 else parsed_outs

    def refine_description(self) -> str:
        """Refine the tool description by replacing the input and output
        markers with raw data types. For example, ```"{{{input: image}}}"```
        will be replaced with ```"image"```.

        Args:
            description (str): The original tool description.

        Returns:
            str: The refined tool description.
        """

        inputs_desc = []
        for in_field, in_category in zip(self.tool.input_fields,
                                         self.toolmeta.inputs):
            agent_type = self.agent_cat2type[in_category]
            inputs_desc.append(f'{in_field} ({in_category} {agent_type})')
        inputs_desc = 'Args: ' + ', '.join(inputs_desc)

        description = f'{self.toolmeta.description} {inputs_desc}'

        return description
