from typing import Tuple

from agentlego.types import AudioIO, File, ImageIO, IOType
from .base_parser import BaseParser


class DefaultParser(BaseParser):
    agent_type2format = {
        ImageIO: 'path',
        AudioIO: 'path',
        File: 'path',
    }

    def parse_inputs(self, *args, **kwargs) -> Tuple[tuple, dict]:
        for arg, p in zip(args, self.tool.inputs):
            kwargs[p.name] = arg

        parsed_kwargs = {}
        for k, v in kwargs.items():
            p = self.tool.arguments.get(k)
            if p is None:
                raise TypeError(f'Got unexcepted keyword argument "{k}".')
            if not isinstance(v, p.type):
                tool_input = p.type(v)
            else:
                tool_input = v
            parsed_kwargs[k] = tool_input

        return (), parsed_kwargs

    def parse_outputs(self, outputs):
        if isinstance(outputs, tuple):
            assert len(outputs) == len(self.toolmeta.outputs)
            parsed_outs = []
            for out in outputs:
                format = self.agent_type2format.get(type(out))
                if isinstance(out, IOType) and format:
                    out = out.to(format)
                parsed_outs.append(out)
            parsed_outs = tuple(parsed_outs)
        elif isinstance(outputs, dict):
            parsed_outs = {}
            for k, out in outputs.items():
                format = self.agent_type2format.get(type(out))
                if isinstance(out, IOType) and format:
                    out = out.to(format)
                parsed_outs[k] = out
        else:
            format = self.agent_type2format.get(type(outputs))
            if isinstance(outputs, IOType) and format:
                outputs = outputs.to(format)
            parsed_outs = outputs

        return parsed_outs

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
        for p in self.tool.inputs:
            desc = f'{p.name}'
            format = self.agent_type2format.get(p.type, p.type.__name__)
            if p.description:
                format += f', {p.description}'
            if p.optional:
                format += f'. Optional, Defaults to {p.default}'
            desc += f' ({format})'
            inputs_desc.append(desc)
        if len(inputs_desc) > 0:
            inputs_desc = 'Args: ' + '; '.join(inputs_desc)
        else:
            inputs_desc = 'No argument.'

        outputs_desc = []
        for p in self.tool.outputs:
            format = self.agent_type2format.get(p.type, p.type.__name__)
            if p.name and p.description:
                desc = f'{p.name} ({format}, {p.description})'
            elif p.name:
                desc = f'{p.name} ({format})'
            elif p.description:
                desc = f'{format} ({p.description})'
            else:
                desc = f'{format}'
            outputs_desc.append(desc)
        if len(outputs_desc) > 0:
            outputs_desc = 'Returns: ' + '; '.join(outputs_desc)
        else:
            outputs_desc = 'No returns.'

        description = ''
        if self.toolmeta.description:
            description += f'{self.toolmeta.description}\n'
        description += f'{inputs_desc}\n{outputs_desc}'

        return description
