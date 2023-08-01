# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

import torch
from mmagic.apis import MMagicInferencer
from PIL import Image

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from .base_tool import BaseTool
from .parsers import BaseParser


def load_mmagic_inferencer(model, setting, device):
    if CACHED_TOOLS.get('mmagic_inferencer' + str(setting), None) is not None:
        mmagic_inferencer = \
            CACHED_TOOLS['mmagic_inferencer' + str(setting)][model]
    else:
        mmagic_inferencer = MMagicInferencer(
            model_name=model, model_setting=setting, device=device)
        CACHED_TOOLS['mmagic_inferencer' +
                     str(setting)][model] = mmagic_inferencer
    return mmagic_inferencer


def load_diffusion_inferencer(model, device):
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    from diffusers.pipelines.stable_diffusion import \
        StableDiffusionSafetyChecker
    if CACHED_TOOLS.get('diffusion_inferencer', None) is not None:
        diffusion_inferencer = CACHED_TOOLS['diffusion_inferencer'][model]
    else:
        torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            model, torch_dtype=torch_dtype)
        diffusion_inferencer = StableDiffusionControlNetPipeline.from_pretrained(  # noqa
            'runwayml/stable-diffusion-v1-5',
            controlnet=controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                'CompVis/stable-diffusion-safety-checker'),
            torch_dtype=torch_dtype)
        diffusion_inferencer.scheduler = UniPCMultistepScheduler.from_config(
            diffusion_inferencer.scheduler.config)
        diffusion_inferencer.to(device)
        CACHED_TOOLS['diffusion_inferencer'][model] = diffusion_inferencer
    return diffusion_inferencer


class Text2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image From User Input Text',
        model={'model': 'stable_diffusion'},
        description='This is a useful tool when you want to generate an image '
        'from a user input text and save it to a file.like: generate an image '
        'of an object or something, or generate an image that includes'
        'some objects. The input to this tool should be an {{{input:text}}} '
        'representing the object description. It returns a {{{output:image}}} '
        'representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.aux_prompt = 'best quality, extremely detailed'
        self._inferencer = load_mmagic_inferencer(self.toolmeta.model['model'],
                                                  None, self.device)

    def apply(self, text: str) -> str:
        text += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_path(
                'image/sd-res.png', func_name='generate-image')
            self._inferencer.infer(text=text, result_out_dir=output_path)
            return output_path


class Seg2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Segmentations',
        model={
            'model_name': 'controlnet',
            'model_setting': 3
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a segmentation image and the user description. like: '
        'generate a real image of a object or something from this segmentation'
        'image. The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_mmagic_inferencer(
            self.toolmeta.model['model_name'],
            self.toolmeta.model['model_setting'], self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_path(
                'image/controlnet-res.png',
                func_name='generate-image-from-seg')
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path


class Canny2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Canny Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 1
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a canny image and the user description. like: '
        'generate a real image of a object or something from this canny image.'
        'The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_mmagic_inferencer(
            self.toolmeta.model['model_name'],
            self.toolmeta.model['model_setting'], self.device)

    def apply(self, image_path: str, text: str) -> str:
        output_path = get_new_image_path(
            'image/controlnet-res.png', func_name='generate-image-from-canny')

        if self.remote:
            from openxlab.model import inference
            out = inference('mmagic/controlnet_canny', [image_path, text])
            with open(output_path, 'wb') as file:
                file.write(out)

        else:
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path


class Pose2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Pose Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 2
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a human pose image and the user description. like: '
        'generate a real image of a human from this human pose image. or '
        'generate a real image of a human from this pose. The input to this '
        'tool should be an {{{input:image}}} and a {{{input:text}}} '
        'representing the image and the text description. It returns a '
        '{{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_mmagic_inferencer(
            self.toolmeta.model['model_name'],
            self.toolmeta.model['model_setting'], self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_path(
                'image/controlnet-res.png',
                func_name='generate-image-from-pose')
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path


class ScribbleText2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Scribble Image',
        model={
            'model_name': 'fusing/stable-diffusion-v1-5-controlnet-scribble',
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a scribble image and the user description. like: '
        'generate a real image of a object or something from this scribble '
        'image. The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.pipe = load_diffusion_inferencer(
            self.toolmeta.model['model_name'], self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(image_path)
            prompt = f'{text}, {self.a_prompt}'
            image = self.pipe(
                prompt,
                image,
                num_inference_steps=20,
                eta=0.0,
                negative_prompt=self.n_prompt,
                guidance_scale=9.0).images[0]
            output_path = get_new_image_path(
                image_path, func_name='generate-image-from-scribble')
            image.save(output_path)
        return output_path


class DepthText2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Depth Image',
        model={},
        description='This is a useful tool when you want to generate a new '
        'real image from a depth image and the user description. like: '
        'generate a real image of a object or something from this depth '
        'image. The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.pipe = load_diffusion_inferencer(
            'fusing/stable-diffusion-v1-5-controlnet-depth', self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(image_path)
            prompt = f'{text}, {self.a_prompt}'
            image = self.pipe(
                prompt,
                image,
                num_inference_steps=20,
                eta=0.0,
                negative_prompt=self.n_prompt,
                guidance_scale=9.0).images[0]
            output_path = get_new_image_path(
                image_path, func_name='generate-image-from-depth')
            image.save(output_path)
        return output_path
