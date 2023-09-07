# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class TextToImage(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Image From Text',
        description='This is a useful tool to generate an image from the '
        'input text. The input text should be a series of keywords '
        'separated by comma',
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
        self.aux_prompt = ', best quality, extremely detailed'
        self.model_name = model
        self.device = device

    def setup(self):
        from mmagic.apis import MMagicInferencer
        self._inferencer = load_or_build_object(
            MMagicInferencer,
            model_name=self.model_name,
            model_setting=None,
            extra_parameters=dict(
                negative_prompt='longbody, lowres, bad anatomy, bad hands, '
                'missing fingers, extra digit, fewer digits, cropped, '
                'worst quality, low quality'),
            device=self.device)

    def apply(self, text: str) -> ImageIO:
        text += self.aux_prompt
        image = self._inferencer.infer(text=text)[0]['infer_results']
        return ImageIO(image)
