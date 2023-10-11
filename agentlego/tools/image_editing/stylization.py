# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import torch

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import require
from agentlego.utils.cache import load_or_build_object
from ..base import BaseTool


def load_instruct_pix2pix(model, device):
    from diffusers import (EulerAncestralDiscreteScheduler,
                           StableDiffusionInstructPix2PixPipeline)

    dtype = torch.float16 if 'cuda' in device else torch.float32
    instruct_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)
    instruct_pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(
        instruct_pix2pix.scheduler.config)

    return instruct_pix2pix


class ImageStylization(BaseTool):
    """A tool to stylize an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``diffusers`` repository.
            Defaults to 'timbrooks/instruct-pix2pix'.
        inference_steps (int): The number of inference steps. Defaults to 20.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Image Modification',
        description='This tool can modify the input image according to the '
        'input instruction. Here are some example instructions: '
        '"turn him into cyborg", "add fireworks to the sky", '
        '"make his jacket out of leather".',
        inputs=['image', 'text'],
        outputs=['image'],
    )

    @require('diffusers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'timbrooks/instruct-pix2pix',
                 inference_steps: int = 20,
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model_name = model
        self.inference_steps = inference_steps
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            load_instruct_pix2pix,
            model=self.model_name,
            device=self.device,
        )

    def apply(self, image: ImageIO, instruction: str) -> ImageIO:
        generated_image = self._inferencer(
            instruction,
            image=image.to_pil().convert('RGB'),
            num_inference_steps=self.inference_steps,
            image_guidance_scale=1.).images[0]
        return ImageIO(generated_image)
