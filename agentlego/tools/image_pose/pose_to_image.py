from typing import Tuple

from PIL import Image

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class PoseToImage(BaseTool):
    """A tool to generate image according to a human pose image.

    Args:
        model (str): The pose controlnet model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can generate an image from a human pose '
                    'image and a text.')

    @require('diffusers')
    def __init__(self, model: str = 'sd', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        assert model in ['sd', 'sdxl']
        self.model = model
        self.device = device

    def setup(self):
        if self.model == 'sdxl':
            self.pipe = load_sdxl(
                controlnet='thibaud/controlnet-openpose-sdxl-1.0',
                device=self.device,
            )
            self.canvas_size = 1024
        elif self.model == 'sd':
            self.pipe = load_sd(
                controlnet='lllyasviel/sd-controlnet-openpose',
                device=self.device,
            )
            self.canvas_size = 512
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(
        self,
        image: ImageIO,
        keywords: Annotated[str,
                            Info('A series of English keywords separated by comma.')],
    ) -> ImageIO:
        text = f'{keywords}, {self.a_prompt}'
        width, height = self.get_image_size(image.to_pil(), canvas_size=self.canvas_size)
        image = self.pipe(
            text,
            image=image.to_pil(),
            negative_prompt=self.n_prompt,
            width=width,
            height=height,
        ).images[0]
        return ImageIO(image)

    @staticmethod
    def get_image_size(image: Image.Image, canvas_size=512) -> Tuple[int, int]:
        # The sd canvas size must can be divided by 8.

        aspect_ratio = image.width / image.height
        width = int((canvas_size * canvas_size * aspect_ratio)**0.5)
        height = int(width / aspect_ratio)

        width = width - (width % 8)
        height = height - (height % 8)

        return width, height
