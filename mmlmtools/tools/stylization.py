# Copyright (c) OpenMMLab. All rights reserved.
import torch
from diffusers import EulerAncestralDiscreteScheduler as ea_scheduler
from diffusers import \
    StableDiffusionInstructPix2PixPipeline as sd_instruct_pix2pix
from PIL import Image

from mmlmtools.utils.toolmeta import ToolMeta
from ..utils.file import get_new_image_path
from .base_tool_v1 import BaseToolv1


class InstructPix2PixTool(BaseToolv1):
    DEFAULT_TOOLMETA = dict(
        name='Instruct Image Using Text',
        model={'model_name': 'timbrooks/instruct-pix2pix'},
        description='This is a useful tool '
        'when you want the style of the image to be like the text. '
        'like: make it looks like a painting, or makie it like a robot etc. ',
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image_path of the image to be changed'
        ' and the text description of stylization.',
        output_description='It returns a string as the output, '
        'representing the image_path of the stylized image. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            if 'cuda' in self.device:
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
            self._inferencer = sd_instruct_pix2pix.from_pretrained(
                self.toolmeta.model['model_name'],
                safety_checker=None,
                torch_dtype=self.torch_dtype).to(self.device)
            self._inferencer.scheduler = ea_scheduler.from_config(
                self._inferencer.scheduler.config)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, text = inputs
        if self.remote:
            raise NotImplementedError
        else:
            original_image = Image.open(image_path)
            image = self._inferencer(
                text,
                image=original_image,
                num_inference_steps=40,
                image_guidance_scale=1.2).images[0]
            updated_image_path = get_new_image_path(
                image_path, func_name='stylization')
            image.save(updated_image_path)
        return updated_image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
