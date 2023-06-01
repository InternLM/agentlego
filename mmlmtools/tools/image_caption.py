# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmpretrain.apis import ImageCaptionInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class ImageCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='ImageCaptionTool',
        model='blip-base_3rdparty_caption',
        description='This is a useful tool '
        'when you want to know what is inside the image.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(toolmeta, input_style, output_style, remote, **kwargs)

        self.inferencer = ImageCaptionInferencer(toolmeta.model, device=device)

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
            with Registry('scope').switch_scope_and_registry('mmpretrain'):
                outputs = self.inferencer(inputs)[0]['pred_caption']
        return outputs
