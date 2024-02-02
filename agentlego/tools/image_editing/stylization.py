from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


def load_instruct_pix2pix(model, device):
    import torch
    from diffusers import (EulerAncestralDiscreteScheduler,
                           StableDiffusionInstructPix2PixPipeline)

    dtype = torch.float16 if 'cuda' in device else torch.float32
    instruct_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)
    instruct_pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(
        instruct_pix2pix.scheduler.config)

    return instruct_pix2pix


class ImageStylization(BaseTool):
    """A tool to stylize an image.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``diffusers`` repository.
            Defaults to 'timbrooks/instruct-pix2pix'.
        inference_steps (int): The number of inference steps. Defaults to 20.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can modify the input image according to the '
                    'input instruction. Here are some example instructions: '
                    '"turn him into cyborg", "add fireworks to the sky", '
                    '"make his jacket out of leather".')

    @require('diffusers')
    def __init__(self,
                 model: str = 'timbrooks/instruct-pix2pix',
                 inference_steps: int = 20,
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model_name = model
        self.inference_steps = inference_steps
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            load_instruct_pix2pix,
            model=self.model_name,
            device=self.device,
        )

    def apply(self, image: ImageIO, instruction: str) -> ImageIO:
        generated_image = self._inferencer(
            instruction,
            image=image.to_pil().convert('RGB'),
            num_inference_steps=self.inference_steps,
            image_guidance_scale=1.).images[0]
        return ImageIO(generated_image)
