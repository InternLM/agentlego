# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.apis import MMOCRInferencer

from .base_tool import BaseTool


class OCRTool(BaseTool):

    def __init__(self,
                 model: str = 'svtr-small',
                 checkpoint: str = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.inferencer = MMOCRInferencer(
            det='dbnetpp', rec=model, device=device)

    def apply(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            outputs = self.inferencer(inputs, **kwargs)
        return outputs
