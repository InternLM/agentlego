from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
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

    default_desc = ('A tool to describe a certain part of the input image.')

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

    def apply(self, image: ImageIO, region: str) -> str:
        """ 
        region: (x1, y1, x2, y2)  
        x1, y1: upper left
        x2, y2: lower right
        """
        cropped_image = ImageIO(image.to_pil().crop(eval(region)))
        cropped_image = cropped_image.to_array()[:, :, ::-1]
        return self._inferencer(cropped_image)[0]['pred_caption']
