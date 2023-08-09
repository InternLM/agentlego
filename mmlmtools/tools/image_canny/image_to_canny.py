# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from ..base_tool import BaseTool


class ImageToCanny(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Edge Detection On Image',
        model=None,
        description='This is a useful tool '
        'when you want to detect the edge of the image.'
        'input should be a {{{input: image}}}'
        'output should be a {{{output: image}}}')

    def __init__(self,
                 *args,
                 low_threshold: int = 100,
                 high_threshold: int = 200,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        if self.remote:
            raise NotImplementedError
        else:
            canny = cv2.Canny(inputs, self.low_threshold,
                              self.high_threshold)[:, :, None]
            canny = np.concatenate([canny] * 3, axis=2)
            return canny
