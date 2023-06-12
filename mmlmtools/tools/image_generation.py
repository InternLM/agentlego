# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.apis import MMagicInferencer
from mmengine import Registry
from mmengine.hub import get_model
from PIL import Image

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

    def apply(self, inputs, **kwargs):
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


class Canny2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='Canny2ImageTool',
        model='mmagic::controlnet/controlnet-canny.py',
        description='This is a useful tool '
        'when you want to generate a new real image from a canny image and '
        'the user description. like: generate a real image of a '
        'object or something from this canny image. or generate a '
        'new real image of a object or something from this edge image. ',
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image_path of a canny '
        'image and the text description of objects to generate.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.inferencer = get_model(self.toolmeta.model).to(
                device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            image = Image.open(image_path)
            text = ','.join(splited_inputs[1:])
        return image, text

    def apply(self, inputs, **kwargs):
        image, prompt = inputs
        if self.remote:
            raise NotImplementedError
        else:
            image_path = get_new_image_name(
                'image/controlnet-res.png',
                func_name='generate-image-from-canny')
            with Registry('scope').switch_scope_and_registry('mmagic'):
                output_dict = self.inferencer.infer(prompt, control=image)
                control = output_dict['samples'][0]
                control.save(image_path)
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
