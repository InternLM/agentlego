# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmlmtools.toolmeta import ToolMeta


class BaseTool(metaclass=ABCMeta):
    """"""
    DEFAULT_TOOLMETA = dict(
        tool_name='BaseTool',
        model=None,
        description='This is a tool can do nothing.')

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

    def convert_inputs(self, inputs, **kwargs):
        """"""
        return inputs

    def convert_outputs(self, outputs, **kwargs):
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

    def apply(self, inputs, **kwargs):
        converted_inputs = self.convert_inputs(inputs, **kwargs)
        outputs = self.infer(converted_inputs, **kwargs)
        results = self.convert_outputs(outputs, **kwargs)
        return results

    def inference(self, inputs, **kwargs):
        return self.apply(inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        return self.apply(inputs, **kwargs)
