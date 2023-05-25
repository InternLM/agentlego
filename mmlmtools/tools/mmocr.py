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

    def inference(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            ocr_results = self.inferencer(
                inputs, show=False, **kwargs)['predictions']
            outputs = []
            for x in ocr_results:
                outputs += x['rec_texts']
        return outputs

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'text':
            outputs = ', '.join(outputs)
            return outputs
