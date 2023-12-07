# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import cv2
import numpy as np

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from ..base import BaseTool


class ImageToCanny(BaseTool):
    """A tool to do edge detection by canny algorithm on an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='EdgeDetectionOnImage',
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
