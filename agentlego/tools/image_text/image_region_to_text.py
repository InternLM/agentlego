from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, parse_multi_float, require
from ..base import BaseTool


class ImageRegionDescription(BaseTool):
    """A tool to describe a certain part of the input image.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``MMPreTrain`` repository.
            Defaults to ``blip-base_3rdparty_caption``.
        device (str): The device to load the model. Defaults to 'cpu'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'A tool to describe a certain part of the input image.'

    @require('mmpretrain')
    def __init__(self,
                 model: str = 'blip-base_3rdparty_caption',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import ImageCaptionInferencer
        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                ImageCaptionInferencer,
                model=self.model,
                device=self.device,
            )

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
    ) -> str:
        x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
        cropped_image = image.to_array()[y1:y2, x1:x2, ::-1]
        return self._inferencer(cropped_image)[0]['pred_caption']
