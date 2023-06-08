# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmocr.apis import MMOCRInferencer

from mmlmtools.toolmeta import ToolMeta
from .base_tool import BaseTool


class OCRTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='OCRTool',
        model='svtr-small',
        description='This is a useful tool '
        'when you want to recognize the text from a photo.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.inferencer = MMOCRInferencer(
                det='dbnetpp', rec=self.toolmeta.model, device=self.device)

    def apply(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmocr'):
                ocr_results = self.inferencer(
                    inputs, show=False, **kwargs)['predictions']
            outputs = []
            for x in ocr_results:
                outputs += x['rec_texts']
        return outputs
