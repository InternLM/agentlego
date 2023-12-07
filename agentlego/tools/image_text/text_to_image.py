# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class TextToImage(BaseTool):
    """A tool to generate image according to some keywords.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The stable diffusion model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='TextToImage',
        description='This tool can generate an image according to the '
        'input text. The input text should be a series of keywords '
        'separated by comma, and all keywords must be in English.',
        inputs=['text'],
        outputs=['image'],
    )

    @require('diffusers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'sd',
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        assert model in ['sd', 'sdxl']
        self.model = model
        self.device = device

    def setup(self):
        if self.model == 'sdxl':
            self.pipe = load_sdxl(device=self.device)
        elif self.model == 'sd':
            self.pipe = load_sd(device=self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, text: str) -> ImageIO:
        prompt = f'{text}, {self.a_prompt}'
        image = self.pipe(
            prompt,
            num_inference_steps=30,
            negative_prompt=self.n_prompt,
        ).images[0]
        return ImageIO(image)
