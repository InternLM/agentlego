from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class SemanticSegmentation(BaseTool):
    """A tool to conduct semantic segmentation on an image.

    Args:
        seg_model (str): The model name used to inference. Which can be found
            in the ``MMSegmentation`` repository.
            Defaults to ``mask2former_r50_8xb2-90k_cityscapes-512x1024``.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can segment all items in the input image and '
                    'return a segmentation result image. '
                    'It focus on urban scene images.')

    @require('mmsegmentation')
    def __init__(self,
                 seg_model: str = 'mask2former_r50_8xb2-90k_cityscapes-512x1024',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
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
