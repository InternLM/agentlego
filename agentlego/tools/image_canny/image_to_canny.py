import numpy as np

from agentlego.types import ImageIO
from agentlego.utils import require
from ..base import BaseTool


class ImageToCanny(BaseTool):
    """A tool to do edge detection by canny algorithm on an image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can extract the edge image from an image.'

    @require('opencv-python')
    def __init__(self, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.low_threshold = 100
        self.high_threshold = 200

    def apply(self, image: ImageIO) -> ImageIO:
        import cv2
        canny = cv2.Canny(image.to_array(), self.low_threshold,
                          self.high_threshold)[:, :, None]
        canny = np.concatenate([canny] * 3, axis=2)
        return ImageIO(canny)
