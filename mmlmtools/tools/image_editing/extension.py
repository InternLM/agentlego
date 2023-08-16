# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.toolmeta import ToolMeta
from .replace import load_inpainting
from ..base_tool import BaseTool
from ..image_caption import load_caption_inferencer
from ..parsers import BaseParser


def blend_gt2pt(old_image, new_image, sigma=0.15, steps=100):
    """Blend the ground truth image with the predicted image.

    This function is copied from 'TaskMatrix/visual_chatgpt.py:
    <https://github.com/microsoft/TaskMatrix/blob/main/visual_chatgpt.py>'_.

    Args:
        old_image (PIL.Image.Image): The ground truth image.
        new_image (PIL.Image.Image): The predicted image.
        sigma (float): The sigma of the Gaussian kernel.
        steps (int): The number of steps to blend.

    Returns:
        PIL.Image.Image: The blended image.
    """
    new_size = new_image.size
    old_size = old_image.size
    easy_img = np.array(new_image)
    gt_img_array = np.array(old_image)
    pos_w = (new_size[0] - old_size[0]) // 2
    pos_h = (new_size[1] - old_size[1]) // 2

    kernel_h = cv2.getGaussianKernel(old_size[1], old_size[1] * sigma)
    kernel_w = cv2.getGaussianKernel(old_size[0], old_size[0] * sigma)
    kernel = np.multiply(kernel_h, np.transpose(kernel_w))

    kernel[steps:-steps, steps:-steps] = 1
    kernel[:steps, :steps] = \
        kernel[:steps, :steps] / kernel[steps - 1, steps - 1]
    kernel[:steps, -steps:] = \
        kernel[:steps, -steps:] / kernel[steps - 1, -(steps)]
    kernel[-steps:, :steps] = \
        kernel[-steps:, :steps] / kernel[-steps, steps - 1]
    kernel[-steps:, -steps:] = \
        kernel[-steps:, -steps:] / kernel[-steps, -steps]
    kernel = np.expand_dims(kernel, 2)
    kernel = np.repeat(kernel, 3, 2)

    weight = np.linspace(0, 1, steps)
    top = np.expand_dims(weight, 1)
    top = np.repeat(top, old_size[0] - 2 * steps, 1)
    top = np.expand_dims(top, 2)
    top = np.repeat(top, 3, 2)

    weight = np.linspace(1, 0, steps)
    down = np.expand_dims(weight, 1)
    down = np.repeat(down, old_size[0] - 2 * steps, 1)
    down = np.expand_dims(down, 2)
    down = np.repeat(down, 3, 2)

    weight = np.linspace(0, 1, steps)
    left = np.expand_dims(weight, 0)
    left = np.repeat(left, old_size[1] - 2 * steps, 0)
    left = np.expand_dims(left, 2)
    left = np.repeat(left, 3, 2)

    weight = np.linspace(1, 0, steps)
    right = np.expand_dims(weight, 0)
    right = np.repeat(right, old_size[1] - 2 * steps, 0)
    right = np.expand_dims(right, 2)
    right = np.repeat(right, 3, 2)

    kernel[:steps, steps:-steps] = top
    kernel[-steps:, steps:-steps] = down
    kernel[steps:-steps, :steps] = left
    kernel[steps:-steps, -steps:] = right

    pt_gt_img = easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]]
    gaussian_gt_img = \
        kernel * gt_img_array + (1 - kernel) * pt_gt_img
    gaussian_gt_img = gaussian_gt_img.astype(np.int64)
    easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]] = \
        gaussian_gt_img
    gaussian_img = Image.fromarray(easy_img)
    return gaussian_img


class ImageExtension(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Image Extension Tool',
        model={'model': 'blip-base_3rdparty_caption'},
        description='This is a useful tool when you want to extend the '
        'picture into a larger image. like: extend the image into a '
        'resolution of 1000x1000. Attention: you must let the image '
        'to be a larger image.The input to this tool should be an '
        '{{{input:image}}} representing the image to be extended, '
        'and a {{{input:text}}} representing resolution of widthxheight.'
        'It returns an {{{output:image}}} representing the extended image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.ImageCaption = load_caption_inferencer(
            self.toolmeta.model['model'], self.device)

        self.inpainting = load_inpainting(self.device)

        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'\
                        'bad lighting, bad background, bad color, '\
                        'bad aliasing, bad distortion, bad motion blur '\
                        'bad consistency with the background '

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            width, height = text.split('x')
            tosize = (int(width), int(height))
            out_painted_image = self.dowhile(image_path, tosize, 4)
            output_path = get_new_image_path(image_path, 'extension')
            out_painted_image.save(output_path)
        return output_path

    def get_BLIP_caption(self, image_path):
        BLIP_caption = self.ImageCaption(image_path)[0]['pred_caption']
        return BLIP_caption

    def resize_image(self, image, max_size=1000000, multiple=8):
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(math.sqrt(max_size * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
        new_width, new_height = new_width - (new_width % multiple),\
            new_height - (new_height % multiple)
        return image.resize((new_width, new_height))

    def dowhile(self, original_img_path, tosize, expand_ratio):
        old_img = Image.open(original_img_path)
        # old_img = ImageOps.crop(old_img, (10, 10, 10, 10))
        while (old_img.size != tosize):
            prompt = self.get_BLIP_caption(original_img_path)
            crop_w = 15 if old_img.size[0] != tosize[0] else 0
            crop_h = 15 if old_img.size[1] != tosize[1] else 0
            old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))
            temp_canvas_size = (expand_ratio * old_img.width if expand_ratio *
                                old_img.width < tosize[0] else tosize[0],
                                expand_ratio * old_img.height if expand_ratio *
                                old_img.height < tosize[1] else tosize[1])
            temp_canvas, temp_mask = Image.new(
                'RGB', temp_canvas_size, color='white'), Image.new(
                    'L', temp_canvas_size, color='white')
            x, y = (temp_canvas.width - old_img.width) // 2, (
                temp_canvas.height - old_img.height) // 2
            temp_canvas.paste(old_img, (x, y))
            temp_mask.paste(0, (x, y, x + old_img.width, y + old_img.height))
            resized_temp_canvas, resized_temp_mask = \
                self.resize_image(temp_canvas), self.resize_image(temp_mask)
            image = self.inpainting(
                prompt=prompt,
                image=resized_temp_canvas,
                mask_image=resized_temp_mask,
                height=resized_temp_canvas.height,
                width=resized_temp_canvas.width,
                num_inference_steps=10).resize(
                    (temp_canvas.width, temp_canvas.height), Image.ANTIALIAS)
            image = blend_gt2pt(old_img, image)
            old_img = image
        return old_img
