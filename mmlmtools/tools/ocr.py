# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.apis import MMOCRInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class OCRTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Recognize the Optical Characters On Image',
        model={
            'det': 'dbnetpp',
            'rec': 'svtr-small'
        },
        description='This is a useful tool '
        'when you want to recognize the text from a photo.',
        input_description='It takes a string as the input, '
        'representing the image_path. ',
        output_description='It returns a string as the output, '
        'representing the text contains the description. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = MMOCRInferencer(
                det=self.toolmeta.model['det'],
                rec=self.toolmeta.model['rec'],
                device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            ocr_results = self._inferencer(inputs, show=False)['predictions']
            outputs = []
            for x in ocr_results:
                outputs += x['rec_texts']
        return outputs

    def convert_outputs(self, outputs):
        if self.output_style == 'text':
            outputs = ', '.join(outputs)
            return outputs
        else:
            raise NotImplementedError
