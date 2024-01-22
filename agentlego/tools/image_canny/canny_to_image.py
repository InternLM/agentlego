from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class CannyTextToImage(BaseTool):
    """A tool to generate image according to a canny edge image.

    Args:
        model (str): The canny controlnet model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can generate an image from a canny edge '
                    'image and keywords.')

    @require('diffusers')
    def __init__(self, model: str = 'sd', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
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
        keywords: Annotated[str,
                            Info('A series of English keywords separated by comma.')],
    ) -> ImageIO:
        prompt = f'{keywords}, {self.a_prompt}'
        image = self.pipe(
            prompt,
            image=image.to_pil(),
            num_inference_steps=20,
            negative_prompt=self.n_prompt,
            controlnet_conditioning_scale=0.5,
        ).images[0]
        return ImageIO(image)
