# Copyright (c) OpenMMLab. All rights reserved.
def load_diffusion_inferencer(model, device):
    """Load the diffusion inferencer.

    Args:
        model (str): The name of the model.
        device (str): The device to use.

    Returns:
        diffusion_inferencer (StableDiffusionControlNetPipeline): The diffusion
            inferencer.
    """

    import torch
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    from diffusers.pipelines.stable_diffusion import \
        StableDiffusionSafetyChecker

    dtype = torch.float16 if 'cuda' in device else torch.float32
    controlnet = ControlNetModel.from_pretrained(model, torch_dtype=dtype)
    diffusion_inferencer = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        controlnet=controlnet,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker'),
        torch_dtype=dtype,
    )
    diffusion_inferencer.scheduler = UniPCMultistepScheduler.from_config(
        diffusion_inferencer.scheduler.config)
    return diffusion_inferencer.to(device)
