import numpy as np

from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToDepth(BaseTool):
    """A tool to estimation depth of an image.

    Args:
        device (str): The device to load the model. Defaults to 'cuda'.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can generate the depth image of an image.'

    @require('transformers')
    def __init__(self, device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
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
