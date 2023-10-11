# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class TextToImage(BaseTool):
    """A tool to generate image according to some keywords.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``MMagic`` repository.
            Defaults to `stable_diffusion`.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Image From Text',
        description='This tool can generate an image according to the '
        'input text. The input text should be a series of keywords '
        'separated by comma, and all keywords must be in English.',
        inputs=['text'],
        outputs=['image'],
    )

    @require('albumentations')
    @require('mmagic')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'stable_diffusion',
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.aux_prompt = (
            'best quality, extremely detailed, master piece, perfect, 4k')
        self.negative_prompt = (
            'longbody, lowres, bad anatomy, bad hands, '
            'missing fingers, extra digit, fewer digits, cropped, '
            'worst quality, low quality')
        self.model_name = model
        self.device = device

    def setup(self):
        from mmagic.apis import MMagicInferencer
        self._inferencer = load_or_build_object(
            MMagicInferencer,
            model_name=self.model_name,
            model_setting=None,
            device=self.device)

    def apply(self, text: str) -> ImageIO:
        text += ', ' + self.aux_prompt
        image = self._inferencer.infer(
            text=text,
            negative_prompt=self.negative_prompt)[0]['infer_results']
        return ImageIO(image)
