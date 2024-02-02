from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class TextToImage(BaseTool):
    """A tool to generate image according to some keywords.

    Args:
        model (str): The stable diffusion model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can generate an image according to the '
                    'input text.')

    @require('diffusers')
    def __init__(self, model: str = 'sd', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        assert model in ['sd', 'sdxl']
        self.model = model
        self.device = device

    def setup(self):
        if self.model == 'sdxl':
            self.pipe = load_sdxl(device=self.device)
        elif self.model == 'sd':
            self.pipe = load_sd(device=self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(
        self,
        keywords: Annotated[str,
                            Info('A series of English keywords separated by comma.')],
    ) -> ImageIO:
        prompt = f'{keywords}, {self.a_prompt}'
        image = self.pipe(
            prompt,
            num_inference_steps=30,
            negative_prompt=self.n_prompt,
        ).images[0]
        return ImageIO(image)
