import numpy as np

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class ReplaceBackgroundOrForeground(BaseTool):
    """A tool to replace the image background or foreground with a new one.

    Args:
        model (str): The canny controlnet model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('The tool can replace the image background or foreground'
                    'with a new one depicted with some keywords.')

    @require(['diffusers', 'opencv-python'])
    def __init__(self, model: str = 'sd', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.low_threshold = 100
        self.high_threshold = 200
        assert model in ['sd', 'sdxl']
        self.model = model
        self.device = device

    def setup(self):
        if self.model == 'sdxl':
            self.pipe = load_sdxl(
                controlnet='diffusers/controlnet-canny-sdxl-1.0',
                controlnet_variant='fp16',
                device=self.device,
            )
        elif self.model == 'sd':
            self.pipe = load_sd(
                controlnet='lllyasviel/sd-controlnet-canny',
                device=self.device,
            )
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(
        self,
        image: ImageIO,
        background: Annotated[str, Info('description of the background')],
        foreground: Annotated[str, Info('description of the foreground')],
    ) -> ImageIO:
        import cv2
        canny = cv2.Canny(image.to_array(), self.low_threshold,
                          self.high_threshold)[:, :, None]
        canny = np.concatenate([canny] * 3, axis=2)

        prompt = (f'background is {background}, '
                  f'foreground is {foreground}, '
                  '{self.a_prompt}')
        image = self.pipe(
            prompt,
            image=ImageIO(canny).to_pil(),
            num_inference_steps=20,
            negative_prompt=self.n_prompt,
            controlnet_conditioning_scale=0.5,
        ).images[0]
        return ImageIO(image)
