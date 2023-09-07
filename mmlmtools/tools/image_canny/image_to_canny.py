# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import cv2
import numpy as np

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from ..base import BaseTool


class ImageToCanny(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Edge Detection On Image',
        description='This tool can extract the edge image from an image.',
        inputs=['image'],
        outputs=['image'],
    )

    def __init__(
        self,
        toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
        parser: Callable = DefaultParser,
    ):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.low_threshold = 100
        self.high_threshold = 200

    def apply(self, image: ImageIO) -> ImageIO:
        canny = cv2.Canny(image.to_array(), self.low_threshold,
                          self.high_threshold)[:, :, None]
        canny = np.concatenate([canny] * 3, axis=2)
        return ImageIO(canny)
