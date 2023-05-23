# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain import ImageCaptionInferencer

from .base_tool import BaseTool


class ImageCaptionTool(BaseTool):

    def __init__(self,
                 model: str = 'blip-base_3rdparty_caption',
                 checkpoint: str = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.inferencer = ImageCaptionInferencer(model, device=device)

    def inference(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            outputs = self.inferencer(inputs)[0]['pred_caption']
        return outputs
