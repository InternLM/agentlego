# Copyright (c) OpenMMLab.All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import require, load_or_build_object
from ..base import BaseTool


class CannyTextToImage(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Image Condition On Canny Image',
        description='This tool can generate an image from a '
        'canny edge image and a description.',
        inputs=['image', 'text'],
        outputs=['image'],
    )

    @require('mmagic')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'controlnet',
                 model_setting: int = 1,
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
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
        res = self._inferencer.infer(text=text, control=image.to_path())
        generated = res[0]['infer_results']
        return ImageIO(generated)
