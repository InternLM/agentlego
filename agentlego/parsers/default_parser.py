from typing import Tuple

from agentlego.types import CatgoryToIO, IOType
from .base_parser import BaseParser


class DefaultParser(BaseParser):
    agent_cat2type = {
        'image': 'path',
        'text': 'string',
        'audio': 'path',
        'int': 'int',
        'bool': 'bool',
        'float': 'float',
    }

    def parse_inputs(self, *args, **kwargs) -> Tuple[tuple, dict]:
        for arg, arg_name in zip(args, self.tool.parameters):
            kwargs[arg_name] = arg

        parsed_kwargs = {}
        for k, v in kwargs.items():
            if k not in self.tool.parameters:
                raise TypeError(f'Got unexcepted keyword argument "{k}".')
            p = self.tool.parameters[k]
            tool_type = CatgoryToIO[p.category]
            if not isinstance(v, tool_type):
                tool_input = tool_type(v)
            else:
                tool_input = v
            parsed_kwargs[k] = tool_input

        return (), parsed_kwargs

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

        return parsed_outs[0] if len(parsed_outs) == 1 else tuple(parsed_outs)

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
        for p in self.tool.parameters.values():
            type_ = self.agent_cat2type[p.category]
            default = f', Defaults to {p.default}' if p.optional else ''
            if p.category in ['image', 'audio']:
                type_ = f'{p.category} {type_}'
            inputs_desc.append(f'{p.name} ({type_}{default})')
        inputs_desc = 'Args: ' + ', '.join(inputs_desc)

        description = f'{self.toolmeta.description} {inputs_desc}'

        return description
