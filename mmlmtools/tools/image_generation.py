# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

import torch
from mmagic.apis import MMagicInferencer
from PIL import Image

from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import get_new_image_name
from .base_tool import BaseTool
from .parsers import BaseParser


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
        self._inferencer = MMagicInferencer(
            model_name=self.toolmeta.model['model'], device=self.device)

    def apply(self, text: str) -> str:
        text += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_name(
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
        self._inferencer = MMagicInferencer(
            model_name=self.toolmeta.model['model_name'],
            model_setting=self.toolmeta.model['model_setting'],
            device=self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_name(
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
        self._inferencer = MMagicInferencer(
            model_name=self.toolmeta.model['model_name'],
            model_setting=self.toolmeta.model['model_setting'],
            device=self.device)

    def apply(self, image_path: str, text: str) -> str:
        output_path = get_new_image_name(
            'image/controlnet-res.png', func_name='generate-image-from-canny')

        if self.remote:
            raise NotImplementedError

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
        self._inferencer = MMagicInferencer(
            model_name=self.toolmeta.model['model_name'],
            model_setting=self.toolmeta.model['model_setting'],
            device=self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_name(
                'image/controlnet-res.png',
                func_name='generate-image-from-pose')
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path


class ScribbleText2Image(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Scribble Image',
        model={},
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
        from diffusers import (ControlNetModel,
                               StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)
        from diffusers.pipelines.stable_diffusion import \
            StableDiffusionSafetyChecker  # noqa
        self.torch_dtype = torch.float16 \
            if 'cuda' in self.device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            'fusing/stable-diffusion-v1-5-controlnet-scribble',
            torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                'CompVis/stable-diffusion-safety-checker'),
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler_config)
        self.pipe.to(self.device)
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
            output_path = get_new_image_name(
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
        from diffusers import (ControlNetModel,
                               StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)
        from diffusers.pipelines.stable_diffusion import \
            StableDiffusionSafetyChecker  # noqa
        self.torch_dtype = torch.float16 \
            if 'cuda' in self.device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            'fusing/stable-diffusion-v1-5-controlnet-depth',
            torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                'CompVis/stable-diffusion-safety-checker'),
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler_config)
        self.pipe.to(self.device)
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
            output_path = get_new_image_name(
                image_path, func_name='generate-image-from-depth')
            image.save(output_path)
        return output_path
