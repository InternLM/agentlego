# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class SemanticSegmentation(BaseTool):
    """A tool to conduct semantic segmentation on an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        seg_model (str): The model name used to inference. Which can be found
            in the ``MMSegmentation`` repository.
            Defaults to ``mask2former_r50_8xb2-90k_cityscapes-512x1024``.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='Semantic Segment the Image',
        description=('This is a useful tool when you only want to segment the '
                     'picture or segment all objects in the picture. like: '
                     'semantic segment all objects in the image. '),
        inputs=['image'],
        outputs=['image'],
    )

    @require('mmsegmentation')
    def __init__(
            self,
            toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
            parser: Callable = DefaultParser,
            seg_model: str = ('mask2former_r50_8xb2-90k_cityscapes-512x1024'),
            device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.seg_model = seg_model
        self.device = device

    def setup(self):
        from mmseg.apis import MMSegInferencer

        self._inferencer = load_or_build_object(
            MMSegInferencer, model=self.seg_model, device=self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        image = image.to_path()
        results = self._inferencer(image, return_vis=True)
        output_image = results['visualization']
        return ImageIO(output_image)
