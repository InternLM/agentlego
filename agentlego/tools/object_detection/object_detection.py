from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ObjectDetection(BaseTool):
    """A tool to detection all objects defined in COCO 80 classes.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``rtmdet_l_8xb32-300e_coco``.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='DetectAllObjects',
        description=('A useful tool when you only want to detect the picture '
                     'or detect all objects in the picture. like: detect all '
                     'objects. '),
        inputs=['image'],
        outputs=['image'],
    )

    @require('mmdet>=3.1.0')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'rtmdet_l_8xb32-300e_coco',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        image = image.to_path()
        results = self._inferencer(image, return_vis=True)
        output_image = results['visualization'][0]
        return ImageIO(output_image)
