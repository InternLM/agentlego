# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

import torch
from PIL import Image

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


def load_diffusion_inferencer(model, device):
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    from diffusers.pipelines.stable_diffusion import \
        StableDiffusionSafetyChecker
    if CACHED_TOOLS.get('diffusion_inferencer', None) is not None:
        diffusion_inferencer = CACHED_TOOLS['diffusion_inferencer'][model]
    else:
        torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            model, torch_dtype=torch_dtype)
        diffusion_inferencer = StableDiffusionControlNetPipeline.from_pretrained(  # noqa
            'runwayml/stable-diffusion-v1-5',
            controlnet=controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                'CompVis/stable-diffusion-safety-checker'),
            torch_dtype=torch_dtype)
        diffusion_inferencer.scheduler = UniPCMultistepScheduler.from_config(
            diffusion_inferencer.scheduler.config)
        diffusion_inferencer.to(device)
        CACHED_TOOLS['diffusion_inferencer'][model] = diffusion_inferencer
    return diffusion_inferencer


class ScribbleTextToImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Scribble Image',
        model={
            'model_name': 'fusing/stable-diffusion-v1-5-controlnet-scribble',
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a scribble image and the user description. like: '
        'generate a real image of a object or something from this scribble '
        'image. The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.pipe = load_diffusion_inferencer(
            self.toolmeta.model['model_name'], self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(image_path)
            prompt = f'{text}, {self.a_prompt}'
            image = self.pipe(
                prompt,
                image,
                num_inference_steps=20,
                eta=0.0,
                negative_prompt=self.n_prompt,
                guidance_scale=9.0).images[0]
            output_path = get_new_image_path(
                image_path, func_name='generate-image-from-scribble')
            image.save(output_path)
        return output_path
