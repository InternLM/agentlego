import math

import numpy as np
from PIL import Image, ImageOps

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, parse_multi_float, require
from ..base import BaseTool
from .replace import Inpainting


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
    import cv2
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


class ImageExpansion(BaseTool):
    """A tool to expand the given image.

    Args:
        caption_model (str): The model name used to inference. Which can be
            found in the ``MMPreTrain`` repository.
            Defaults to ``blip-base_3rdparty_caption``.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can expand the peripheral area of an image '
                    'based on its content, thus obtaining a larger image.')

    @require('mmpretrain')
    @require('diffusers')
    def __init__(self,
                 caption_model: str = 'blip-base_3rdparty_caption',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.caption_model_name = caption_model
        self.device = device

    def setup(self):

        from mmpretrain.apis import ImageCaptionInferencer

        self.caption_inferencer = load_or_build_object(
            ImageCaptionInferencer, model=self.caption_model_name, device=self.device)

        self.inpainting_inferencer = load_or_build_object(Inpainting, device=self.device)

    def apply(
        self,
        image: ImageIO,
        scale: Annotated[str,
                         Info('expand ratio, can be a float number or two '
                              'float number for width and height ratio.')],
    ) -> ImageIO:
        old_img = image.to_pil().convert('RGB')
        expand_ratio = 4  # maximum expand ratio for a single round.

        scale_w, scale_h = parse_multi_float(scale, 2)
        target_w = int(old_img.size[0] * scale_w)
        target_h = int(old_img.size[1] * scale_h)

        while old_img.size != (target_w, target_h):
            caption = self.get_caption(old_img)

            # crop the some border to re-generation.
            crop_w = 15 if (old_img.width != target_w and old_img.width > 100) else 0
            crop_h = 15 if (old_img.height != target_h and old_img.height > 100) else 0
            old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))

            canvas_w = min(expand_ratio * old_img.width, target_w)
            canvas_h = min(expand_ratio * old_img.height, target_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')
            mask = Image.new('L', (canvas_w, canvas_h), color='white')

            # paste the old image into the center of canvas.
            x = (canvas.width - old_img.width) // 2
            y = (canvas.height - old_img.height) // 2
            canvas.paste(old_img, (x, y))
            mask.paste(0, (x, y, x + old_img.width, y + old_img.height))

            # Resize the canvas into a proper size (about 1000x1000) to
            # generate more details
            resized_canvas = self.resize_image(canvas)
            resized_mask = self.resize_image(mask)
            image = self.inpainting_inferencer(
                prompt=caption,
                image=resized_canvas,
                mask_image=resized_mask,
                height=resized_canvas.height,
                width=resized_canvas.width,
                num_inference_steps=10)

            # Resize the generated image into the canvas size and
            # blend with the old image.
            image = image.resize((canvas.width, canvas.height), Image.ANTIALIAS)
            image = blend_gt2pt(old_img, image)
            old_img = image

        return ImageIO(old_img)

    def get_caption(self, image: Image.Image):
        image = np.array(image)[:, :, ::-1]
        return self.caption_inferencer(image)[0]['pred_caption']

    def resize_image(self, image, max_size=1000000, multiple=8):
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(math.sqrt(max_size * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
        new_width = new_width - (new_width % multiple)
        new_height = new_height - (new_height % multiple)
        return image.resize((new_width, new_height))
