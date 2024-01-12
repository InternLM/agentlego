from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ImageToScribble(BaseTool):
    """A tool to convert image to a scribble sketch.

    Args:
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can generate a sketch scribble of an image.'

    @require('controlnet_aux')
    def __init__(self, device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.device = device

    def setup(self):
        from controlnet_aux import HEDdetector
        self.detector = load_or_build_object(
            HEDdetector.from_pretrained,
            'lllyasviel/Annotators',
        ).to(self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        scribble = self.detector(image.to_pil(), scribble=True)
        return ImageIO(scribble)
