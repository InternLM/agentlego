# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from PIL import Image

from mmlmtools.utils.toolmeta import ToolMeta
from ..utils.file import get_new_image_path
from .base_tool import BaseTool
from .parsers import BaseParser


class InstructPix2PixToolv2(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Instruct Image Using Text',
        model={'model': 'timbrooks/instruct-pix2pix'},
        description='This is a useful tool '
        'when you want the style of the image to be like the text. '
        'like: make it looks like a painting, or makie it like a robot etc. '
        'The input to this tool should be an {{{input:image}}} and '
        'an {{{input:text}}} representing the text description of stylization.'
        'It returns an {{{output:image}}} representing the stylized image. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        from diffusers import EulerAncestralDiscreteScheduler as ea_scheduler
        from diffusers import \
            StableDiffusionInstructPix2PixPipeline as sd_instruct_pix2pix
        if 'cuda' in self.device:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self._inferencer = sd_instruct_pix2pix.from_pretrained(
            self.toolmeta.model['model'],
            safety_checker=None,
            torch_dtype=torch_dtype).to(self.device)
        self._inferencer.scheduler = ea_scheduler.from_config(
            self._inferencer.scheduler.config)

    def apply(self, image_path: str, text: str) -> str:
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