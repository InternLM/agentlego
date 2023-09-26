# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Tuple, Union

from PIL import Image

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class PoseToImage(BaseTool):
    """A tool to generate image according to a human pose image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``MMagic`` repository.
            Defaults to `controlnet`.
        model_setting (int): The index of the model settings. Defaults to 2.
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

    @require('mmagic')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'controlnet',
                 model_setting: int = 2,
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.aux_prompt = (
            'best quality, extremely detailed, master piece, perfect, 4k')
        self.negative_prompt = (
            'longbody, lowres, bad anatomy, bad hands, '
            'missing fingers, extra digit, fewer digits, cropped, '
            'worst quality, low quality')
        self.model_name = model
        self.model_setting = model_setting
        self.device = device

    def setup(self):
        from mmagic.apis import MMagicInferencer

        self._inferencer = load_or_build_object(
            MMagicInferencer,
            model_name=self.model_name,
            model_setting=self.model_setting,
            device=self.device,
        )

    def apply(self, image: ImageIO, text: str) -> ImageIO:
        text = f'{text}, {self.aux_prompt}'
        width, height = self.get_image_size(image.to_pil())
        res = self._inferencer.infer(
            text=text,
            negative_prompt=self.negative_prompt,
            control=image.to_path(),
            extra_parameters=dict(height=height, width=width))
        generated = res[0]['infer_results']
        return ImageIO(generated)

    @staticmethod
    def get_image_size(image: Image.Image, canvas_size=512) -> Tuple[int, int]:
        # The sd canvas size must can be divided by 8.

        aspect_ratio = image.width / image.height
        width = int((canvas_size * canvas_size * aspect_ratio)**0.5)
        height = int(width / aspect_ratio)

        width = width - (width % 8)
        height = height - (height % 8)

        return width, height
