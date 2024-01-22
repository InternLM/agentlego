from typing import Optional

from agentlego.utils import load_or_build_object


def load_sd(model: str = 'runwayml/stable-diffusion-v1-5',
            variant: Optional[str] = 'fp16',
            vae: Optional[str] = None,
            vae_variant: Optional[str] = None,
            controlnet: Optional[str] = None,
            controlnet_variant: Optional[str] = None,
            device=None):
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionControlNetPipeline, StableDiffusionPipeline)

    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    params = {'torch_dtype': dtype}

    if variant is not None:
        params['variant'] = variant
    if vae is not None:
        vae = load_or_build_object(
            AutoencoderKL.from_pretrained,
            vae,
            torch_dtype=dtype,
            variant=vae_variant,
        )
        params['vae'] = vae

    t2i = load_or_build_object(StableDiffusionPipeline.from_pretrained, model, **params)

    if controlnet is None:
        return t2i.to(device)
    else:
        controlnet = load_or_build_object(
            ControlNetModel.from_pretrained,
            controlnet,
            torch_dtype=dtype,
            variant=controlnet_variant,
        )
        pipe = StableDiffusionControlNetPipeline(**t2i.components, controlnet=controlnet)
        return pipe.to(device)


def load_sdxl(model: str = 'stabilityai/stable-diffusion-xl-base-1.0',
              variant: Optional[str] = 'fp16',
              vae: Optional[str] = 'madebyollin/sdxl-vae-fp16-fix',
              vae_variant: Optional[str] = None,
              controlnet: Optional[str] = None,
              controlnet_variant: Optional[str] = None,
              device=None):
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionXLControlNetPipeline,
                           StableDiffusionXLPipeline)
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    params = {'torch_dtype': dtype}

    if variant is not None:
        params['variant'] = variant
    if vae is not None:
        vae = load_or_build_object(
            AutoencoderKL.from_pretrained,
            vae,
            torch_dtype=dtype,
            variant=vae_variant,
        )
        params['vae'] = vae

    t2i = load_or_build_object(StableDiffusionXLPipeline.from_pretrained, model,
                               **params)

    if controlnet is None:
        return t2i.to(device)
    else:
        controlnet = load_or_build_object(
            ControlNetModel.from_pretrained,
            controlnet,
            torch_dtype=dtype,
            variant=controlnet_variant,
        )
        pipe = StableDiffusionXLControlNetPipeline(
            **t2i.components, controlnet=controlnet)
        return pipe.to(device)
