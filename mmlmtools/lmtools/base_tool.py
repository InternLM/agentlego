# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseToolModule


class BaseTool(BaseToolModule):
    """"""

    def __init__(self,
                 model: str = None,
                 checkpoint: str = None,
                 input_type: str = None,
                 output_type: str = None,
                 remote: bool = False,
                 **kwargs):
        self.model = model
        self.checkpoint = checkpoint
        self.input_type = input_type
        self.output_type = output_type
        self.remote = remote

    def convert_inputs(self, inputs, **kwargs):
        return inputs

    def convert_outputs(self, outputs, **kwargs):
        return outputs

    def apply(self, inputs, **kwargs):
        return inputs
