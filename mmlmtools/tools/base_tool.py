# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from .base import BaseToolModule


class BaseTool(BaseToolModule):
    """"""

    def __init__(self,
                 model: str = None,
                 checkpoint: str = None,
                 input_style: str = None,
                 output_style: str = None,
                 remote: bool = False,
                 device: str = 'cpu',
                 **kwargs):
        self.model = model
        self.checkpoint = checkpoint
        self.input_style = input_style
        self.output_style = output_style
        self.remote = remote
        self.device = device

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
