# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool
from ..utils.diffusers import load_diffusion_inferencer


class DepthTextToImage(BaseTool):
    """A tool to generate image according to a depth image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``diffusers`` repository.
            Defaults to 'lllyasviel/sd-controlnet-depth'.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Image Condition On Depth Image',
        description='This tool can generate an image from a depth '
        'image and a text. The text should be a series of English keywords '
        'separated by comma.',
        inputs=['image', 'text'],
        outputs=['image'],
    )

    @require('diffusers')
    def __init__(
            self,
            toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
            parser: Callable = DefaultParser,
            model: str = 'lllyasviel/sd-controlnet-depth',
            device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model_name = model
        self.device = device

    def setup(self):
        self.pipe = load_or_build_object(
            load_diffusion_inferencer,
            self.model_name,
            self.device,
        )
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, image: ImageIO, text: str) -> ImageIO:
        prompt = f'{text}, {self.a_prompt}'
        image = self.pipe(
            prompt,
            image.to_pil(),
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0).images[0]
        return ImageIO(image)
