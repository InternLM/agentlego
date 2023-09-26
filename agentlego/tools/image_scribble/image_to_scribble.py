# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToScribble(BaseTool):
    """A tool to convert image to a scribble sketch.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

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
