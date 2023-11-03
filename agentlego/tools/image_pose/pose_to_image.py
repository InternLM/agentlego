# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Tuple, Union

from PIL import Image

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class PoseToImage(BaseTool):
    """A tool to generate image according to a human pose image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The pose controlnet model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Image Condition On Pose Image',
        description='This tool can generate an image from a human pose '
        'image and a text. The text should be a series of English keywords '
        'separated by comma.',
        inputs=['image', 'text'],
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
            self.pipe = load_sdxl(
                controlnet='thibaud/controlnet-openpose-sdxl-1.0',
                device=self.device,
            )
            self.canvas_size = 1024
        elif self.model == 'sd':
            self.pipe = load_sd(
                controlnet='lllyasviel/sd-controlnet-openpose',
                device=self.device,
            )
            self.canvas_size = 512
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, image: ImageIO, text: str) -> ImageIO:
        text = f'{text}, {self.a_prompt}'
        width, height = self.get_image_size(
            image.to_pil(), canvas_size=self.canvas_size)
        image = self.pipe(
            text,
            image=image.to_pil(),
            negative_prompt=self.n_prompt,
            width=width,
            height=height,
        ).images[0]
        return ImageIO(image)

    @staticmethod
    def get_image_size(image: Image.Image, canvas_size=512) -> Tuple[int, int]:
        # The sd canvas size must can be divided by 8.

        aspect_ratio = image.width / image.height
        width = int((canvas_size * canvas_size * aspect_ratio)**0.5)
        height = int(width / aspect_ratio)

        width = width - (width % 8)
        height = height - (height % 8)

        return width, height
