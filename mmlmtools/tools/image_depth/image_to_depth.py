# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import numpy as np

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToDepth(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Generate Depth Image On Image',
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
