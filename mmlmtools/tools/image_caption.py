# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.apis import ImageCaptionInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class ImageCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Get Photo Description',
        model={'model': 'blip-base_3rdparty_caption'},
        description='This is a useful tool '
        'when you want to know what is inside the image.',
        input_description='It takes a string as the input, '
        'representing the text that the tool required. ',
        output_description='It returns a string as the output, '
        'representing the text contains the description. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = ImageCaptionInferencer(
                model=self.toolmeta.model['model'], device=self.device)

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
            import json

            from openxlab.model.handler.model_inference import inference

            out = inference('mmpretrain/blip', ['./demo_text_ocr.jpg'])
            print('json result: {0}'.format(json.loads(out)))

            raise NotImplementedError
        else:
            outputs = self._inferencer(inputs)[0]['pred_caption']
        return outputs
