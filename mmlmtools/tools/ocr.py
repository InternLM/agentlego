# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmocr.apis import MMOCRInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
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
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(toolmeta, input_style, output_style, remote, **kwargs)

        self.inferencer = MMOCRInferencer(
            det='dbnetpp', rec=toolmeta.model, device=device)

    def convert_inputs(self, inputs, **kwargs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def infer(self, inputs, **kwargs):
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

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'text':
            outputs = ', '.join(outputs)
            return outputs
        else:
            raise NotImplementedError
