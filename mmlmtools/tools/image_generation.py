# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.apis import MMagicInferencer
from mmengine import Registry

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Text2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='Text2ImageTool',
        model='stable_diffusion',
        description='This is a useful tool '
        'when you want to generate an image from'
        'a user input text and save it to a file. like: generate '
        'an image of an object or something, or generate an image that includes some objects.'  # noqa
    )

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.aux_prompt = 'best quality, extremely detailed'
            self.inferencer = MMagicInferencer(
                model_name=self.toolmeta.model, device=self.device)

    def infer(self, inputs, **kwargs):
        inputs += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            image_path = get_new_image_name(
                'image/sd-res.png', func_name='generate-image')
            with Registry('scope').switch_scope_and_registry('mmagic'):
                self.inferencer.infer(
                    text=inputs, result_out_dir=image_path, **kwargs)
        return image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
