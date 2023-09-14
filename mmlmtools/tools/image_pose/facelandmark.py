# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class HumanFaceLandmark(BaseTool):
    """A tool to extract human face landmarks from an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``MMPose`` repository.
            Defaults to 'face'.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Human Face Landmark On Image',
        description='This tool can estimate the landmark or keypoints of '
        'human faces in an image and draw the landmarks image.',
        inputs=['image'],
        outputs=['image'],
    )

    @require('mmpose')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'face',
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model_name = model
        self.device = device

    def setup(self):
        from mmpose.apis import MMPoseInferencer
        self._inferencer = load_or_build_object(
            MMPoseInferencer, pose2d=self.model_name, device=self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        image = image.to_array()[:, :, ::-1]
        results = next(
            self._inferencer(
                inputs=image,
                skeleton_style='mmpose',
                black_background=True,
                return_vis=True,
            ))
        landmarks = results['visualization'][0][:, :, ::-1]
        return ImageIO(landmarks)
