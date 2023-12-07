# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import numpy as np

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToDepth(BaseTool):
    """A tool to estimation depth of an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='ImageToDepth',
        description='This tool can generate the depth image of an image.',
        inputs=['image'],
        outputs=['image'],
    )

    @require('transformers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        from transformers import pipeline
        self.depth_estimator = load_or_build_object(
            pipeline,
            'depth-estimation',
        )

    def apply(self, image: ImageIO) -> ImageIO:
        depth = self.depth_estimator(image.to_pil())['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        return ImageIO(depth)
