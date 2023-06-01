# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmpretrain.apis import ImageCaptionInferencer

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

    def infer(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmpretrain'):
                outputs = self.inferencer(inputs)[0]['pred_caption']
        return outputs
