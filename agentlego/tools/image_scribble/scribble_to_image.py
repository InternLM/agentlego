from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd


class ScribbleTextToImage(BaseTool):
    """A tool to generate image according to a scribble sketch.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``diffusers`` repository.
            Defaults to 'lllyasviel/sd-controlnet_scribble'.
        model (str): The scribble controlnet model to use. You can only choose
            "sd" by now. Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can generate an image from a sketch scribble '
                    'image and a text.')

    @require('diffusers')
    def __init__(self, model: str = 'sd', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        assert model in ['sd']
        self.model_name = model
        self.device = device

    def setup(self):
        if self.model == 'sd':
            self.pipe = load_sd(
                controlnet='lllyasviel/sd-controlnet-scribble',
                device=self.device,
            )
        self.a_prompt = 'best quality, extremely detailed, 4k, master piece'
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
            image.to_pil(),
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        return ImageIO(image)
