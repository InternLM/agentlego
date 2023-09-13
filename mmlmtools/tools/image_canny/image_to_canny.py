# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np

from mmlmtools.parsers import BaseParser
from mmlmtools.schema import ToolMeta
from ..base import BaseTool


class ImageToCanny(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Edge Detection On Image',
        model=None,
        description='This is a useful tool '
        'when you want to detect the edge of the image.'
        'input should be a {{{input:image}}}'
        'output should be a {{{output:image}}}')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):

        super().__init__(toolmeta, parser, remote, device)
        self.low_threshold = 100
        self.high_threshold = 200

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        if self.remote:
            raise NotImplementedError
        else:
            canny = cv2.Canny(inputs, self.low_threshold,
                              self.high_threshold)[:, :, None]
            canny = np.concatenate([canny] * 3, axis=2)
            return canny
