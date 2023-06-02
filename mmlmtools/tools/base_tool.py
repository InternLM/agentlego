# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmlmtools.toolmeta import ToolMeta


class BaseTool(metaclass=ABCMeta):
    """"""
    DEFAULT_TOOLMETA = dict(
        tool_name='BaseTool',
        model=None,
        description='This is an abstract tool interface '
        'with no actual function.',
        input_description=None,
        output_description=None)

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = None,
                 output_style: str = None,
                 remote: bool = False,
                 device: str = 'cpu'):
        self.input_style = input_style
        self.output_style = output_style
        self.remote = remote
        self.device = device
        self.toolmeta = toolmeta if toolmeta else ToolMeta(
            **self.DEFAULT_TOOLMETA)

        if toolmeta is not None:
            self.toolmeta = toolmeta
        else:
            assert hasattr(self, 'DEFAULT_TOOLMETA')
            class_name = self.__class__.__name__
            assert self.DEFAULT_TOOLMETA.tool_name != class_name, (
                'self.DEFAULT_TOOLMETA.tool_name should be the same as '
                'the class name of the tool.')
            assert self.DEFAULT_TOOLMETA.description is not None, (
                'self.DEFAULT_TOOLMETA.description should not be None.')
            self.toolmeta = ToolMeta(**self.DEFAULT_TOOLMETA)

        self.format_description()

    def format_description(self):
        func_descrip = self.toolmeta.description
        input_descrip = self.generate_input_description()
        output_descrip = self.generate_output_description()
        res = f'{func_descrip} {input_descrip} {output_descrip}'
        self.toolmeta.description = res
        return res

    def convert_inputs(self, inputs):
        """"""
        return inputs

    def convert_outputs(self, outputs):
        """"""
        return outputs

    @abstractmethod
    def infer(self, inputs, **kwargs):
        """if self.remote:

        raise NotImplementedError
        else:
            outputs = self.inferencer(inputs)
        return outputs
        """

    @abstractmethod
    def setup(self):
        """instantiate inferencer."""

    def apply(self, inputs, **kwargs):
        self.setup()
        converted_inputs = self.convert_inputs(inputs)
        outputs = self.infer(converted_inputs, **kwargs)
        results = self.convert_outputs(outputs)
        return results

    def inference(self, inputs, **kwargs):
        return self.apply(inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        return self.apply(inputs, **kwargs)

    def generate_input_description(self):
        """generate input description according to input style."""
        if self.toolmeta.input_description is not None:
            return self.toolmeta.input_description

        if self.input_style == 'image_path':
            res = 'It takes a string as the input, representing the image_path. '  # noqa
        elif self.input_style == 'text':
            res = 'It takes a string as the input, representing the text that the tool required. '  # noqa
        elif self.input_style == 'image_path, text':
            res = 'The input to this tool should be a comma separated string of two, representing the image_path and the text description of objects. '  # noqa
        elif self.input_style == 'pil image':
            res = 'It takes a <PIL Image> typed image as the input. '
        else:
            raise NotImplementedError
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
            raise NotImplementedError
        return res
