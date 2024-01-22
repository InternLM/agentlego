import copy

from transformers.tools import Tool
from transformers.tools.agent_types import AgentAudio, AgentImage, AgentText, AgentType

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
        # transformers agent system
        self.name: str = 'agentlego_' + tool.name.lower().replace(' ', '_')

        self.description: str = self.refine_description(tool)

        self.inputs = list(tool.toolmeta.inputs)
        self.outputs = list(tool.toolmeta.outputs)

    def __call__(self, *args, **kwargs):
        for arg, p in zip(args, self.tool.inputs):
            kwargs[p.name] = arg

        parsed_kwargs = {}
        for k, v in kwargs.items():
            p = self.tool.arguments[k]
            if p.type is AudioIO:
                parsed_kwargs[k] = AudioIO(v)
            elif p.type is ImageIO:
                parsed_kwargs[k] = ImageIO(v)
            else:
                parsed_kwargs[k] = p.type(v)

        outputs = self.tool(**parsed_kwargs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        parsed_outs = [cast_lego_to_hf(out) for out in outputs]

        return parsed_outs[0] if len(parsed_outs) == 1 else parsed_outs

    @staticmethod
    def refine_description(tool) -> str:
        inputs_desc = []
        type2format = {ImageIO: 'image', AudioIO: 'audio'}
        for p in tool.inputs:
            desc = f'{p.name}'
            format = type2format.get(p.type, p.type.__name__)
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
        for p in tool.outputs:
            format = type2format.get(p.type, p.type.__name__)
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

        description = (f'{tool.toolmeta.description}\n'
                       f'{inputs_desc}\n{outputs_desc}')

        return description
