# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain import ImageCaptionInferencer

from .base_tool import BaseTool


class ImageCaptionTool(BaseTool):

    def __init__(self,
                 model: str = None,
                 checkpoint: str = None,
                 input_type: str = None,
                 output_type: str = None,
                 remote: bool = False,
                 **kwargs):
        super().__init__(model, checkpoint, input_type, output_type, remote,
                         **kwargs)

        self.inferencer = ImageCaptionInferencer('blip-base_3rdparty_caption')

    def convert_inputs(self, inputs, **kwargs):
        return inputs

    def convert_outputs(self, outputs, **kwargs):
        return outputs

    def inference(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            outputs = self.inferencer(inputs=inputs, **self.call_args)
        return outputs
