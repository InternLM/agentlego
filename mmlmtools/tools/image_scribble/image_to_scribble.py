# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToScribble(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Scribble Conditioned On Image',
        description='This tool can generate a sketch scribble of an image.',
        inputs=['image'],
        outputs=['image'],
    )

    @require('controlnet_aux')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        from controlnet_aux import HEDdetector
        self.detector = load_or_build_object(
            HEDdetector.from_pretrained,
            'lllyasviel/Annotators',
        ).to(self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        scribble = self.detector(image.to_pil(), scribble=True)
        return ImageIO(scribble)
