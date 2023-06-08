# Copyright (c) OpenMMLab. All rights reserved.
import re
from abc import ABCMeta, abstractmethod

from mmengine import is_list_of

from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import inputs_conversions, outputs_conversions


class BaseTool(metaclass=ABCMeta):
    DEFAULT_TOOLMETA = dict(
        tool_name='BaseTool',
        model=None,
        description='This is an abstract tool interface '
        'with no actual function.',
        input_description=None,
        output_description=None)

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'identity',
                 output_style: str = 'identity',
                 remote: bool = False,
                 device: str = 'cpu',
                 **kwargs):
        self.input_style = input_style
        self.output_style = output_style
        self.remote = remote
        self.device = device
        self.init_args = kwargs

        if toolmeta is not None:
            self.toolmeta = toolmeta
        else:
            assert hasattr(self, 'DEFAULT_TOOLMETA')
            assert self.DEFAULT_TOOLMETA.get('description') is not None, (
                '`description` in `DEFAULT_TOOLMETA` should not be None.')
            self.toolmeta = ToolMeta(**self.DEFAULT_TOOLMETA)

        self.format_description()

    def format_description(self):
        """Generate complete description."""
        func_descrip = self.toolmeta.description
        input_descrip = self.generate_input_description()
        output_descrip = self.generate_output_description()
        res = f'{func_descrip} {input_descrip} {output_descrip}'
        self.toolmeta.description = res
        return res

    def convert_inputs(self, inputs):
        return self.convert(inputs, inputs_conversions, self.input_style)

    def convert_outputs(self, outputs):
        return self.convert(outputs, outputs_conversions, self.output_style)

    def convert(self, inputs, mapping, style):
        """Convert inputs into the tool required format."""
        if not self._is_composed_conversion(style):
            return mapping[style](inputs)

        # This is not a general regex for all conditions. However, consider
        # most inputs will not consider more than two element, this
        # regex is enough
        style_pattern = re.sub(r'(\{[^}]*\})', '(.+?)', style) + '$'
        inputs = re.findall(style_pattern, inputs)
        # if inputs have multiple groups, `find_all` will return a list of
        # tuple with all matched groups
        if is_list_of(inputs, tuple):
            inputs = inputs[0]
        func_names = re.findall(r'\{([^}]*)\}', style)

        res = []
        for input, func_name in zip(inputs, func_names):
            res.append(mapping[func_name](input))

        return tuple(res)

    @abstractmethod
    def apply(self, inputs, **kwargs):
        """if self.remote:

        raise NotImplementedError
        else:
            outputs = self.inferencer(inputs)
        return outputs
        """

    @abstractmethod
    def setup(self):
        """instantiate inferencer."""

    def __call__(self, inputs, **kwargs):
        self.setup()
        converted_inputs = self.convert_inputs(inputs)
        if not isinstance(converted_inputs, tuple):
            converted_inputs = (converted_inputs, )
        outputs = self.apply(*converted_inputs, **kwargs)
        outputs = self.convert_outputs(outputs)
        return outputs

    def inference(self, inputs, **kwargs):
        """This method is for compatibility with the LangChain tool
        interface."""
        return self(inputs, **kwargs)

    def generate_input_description(self):
        """generate input description according to input style."""
        if self.toolmeta.input_description is not None:
            return self.toolmeta.input_description

        if self.input_style == 'image_path':
            res = 'It takes a string as the input, representing the image_path. '  # noqa
        elif self.input_style == 'text':
            res = 'It takes a string as the input, representing the text that the tool required. '  # noqa
        elif self.input_style == '{image_path}, {text}':
            res = 'The input to this tool should be a comma separated string of two, representing the image_path and the text description of objects. '  # noqa
        elif self.input_style == 'pil image':
            res = 'It takes a <PIL Image> typed image as the input. '
        elif self.input_style == '{pil image}, {text}':
            res = 'The input to this tool should be a comma separated string of two, representing the <PIL Image> typed image and the text description of objects. '  # noqa
        else:
            res = ''
        return res

    def generate_output_description(self):
        """generate output description according to output style."""
        if self.toolmeta.output_description is not None:
            return self.toolmeta.output_description

        if self.output_style == 'image_path':
            res = 'It returns a string as the output, representing the image_path. '  # noqa
        elif self.output_style == 'text':
            res = 'It returns a string as the output, representing the text contains the description. '  # noqa
        elif self.output_style == 'pil image':
            res = 'It returns a <PIL Image> typed image as the output. '  # noqa
        else:
            res = ''
        return res

    def _is_composed_conversion(self, convert):
        return isinstance(convert, str) and '{' in convert
