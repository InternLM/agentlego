# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser
from ..segment_anything import load_sam_and_predictor

try:
    from mmdet.apis import DetInferencer
    has_mmdet = True
except ImportError:
    has_mmdet = False

GLOBAL_SEED = 1912


def load_grounding(model, device):
    """Load grounding model.

    Args:
        model (str): The name of the model.
        device (str): The device to use.

    Returns:
        grounding (DetInferencer): The grounding model.
    """
    if CACHED_TOOLS.get('grounding', None) is not None:
        grounding = CACHED_TOOLS['grounding']
    else:
        if not has_mmdet:
            raise RuntimeError('mmdet is required but not installed')
        grounding = DetInferencer(model=model, device=device)
        CACHED_TOOLS['grounding'] = grounding
    return grounding


def load_inpainting(device):
    """Load inpainting model.

    Args:
        device (str): The device to use.

    Returns:
        inpainting (Inpainting): The inpainting model.
    """
    if CACHED_TOOLS.get('inpainting', None) is not None:
        inpainting = CACHED_TOOLS['inpainting']
    else:
        inpainting = Inpainting(device)
        CACHED_TOOLS['inpainting'] = inpainting
    return inpainting


class Inpainting:
    """Inpainting model.

    Refers to 'TaskMatrix/visual_chatgpt.py:
    <https://github.com/microsoft/TaskMatrix/blob/main/visual_chatgpt.py>'_.

    Args:
        device (str): The device to use.
    """

    def __init__(self, device):
        from diffusers import StableDiffusionInpaintPipeline

        self.device = device
        self.revision = 'fp16' if 'cuda' in self.device else None
        self.torch_dtype = torch.float16 \
            if 'cuda' in self.device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting',
            revision=self.revision,
            torch_dtype=self.torch_dtype).to(device)
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'\
                        'bad lighting, bad background, bad color, '\
                        'bad aliasing, bad distortion, bad motion blur '\
                        'bad consistency with the background '

    def __call__(self,
                 prompt,
                 image,
                 mask_image,
                 height=512,
                 width=512,
                 num_inference_steps=20):
        update_image = self.inpaint(
            prompt=prompt,
            negative_prompt=self.n_prompt,
            image=image.resize((width, height)),
            mask_image=mask_image.resize((width, height)),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps).images[0]
        return update_image


class ObjectReplace(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Replace the Given Object In The Image',
        model={
            'model': 'sam_vit_h_4b8939.pth',
            'grounding': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'
        },
        description='This is a useful tool when you want to replace the '
        'certain objects in the image with another object. like: replace '
        'the cat in this image with a dog, or can you replace the cat '
        'with a dog for me. The input to this tool should be an '
        '{{{input:image}}} and {{{input:text}}} representing the object '
        'to be replaced and {{{input:text}}} representing the object to '
        'replace with. It returns a {{{output:image}}} representing '
        'the image with the replaced object.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.grounding = load_grounding(self.toolmeta.model['grounding'],
                                        self.device)

        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.toolmeta.model['model'],
            f"model_zoo/{self.toolmeta.model['model']}", True, self.device)

        self.inpainting = load_inpainting(self.device)

    def apply(self, image_path: str, text1: str, text2: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image_pil = Image.open(image_path).convert('RGB')
            results = self.grounding(
                inputs=image_path,
                texts=[text1],
                no_save_vis=True,
                return_datasample=True)
            results = results['predictions'][0].pred_instances

            boxes_filt = results.bboxes

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image)
            masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)
            mask = torch.sum(masks, dim=0).unsqueeze(0)
            mask = torch.where(mask > 0, True, False)
            mask = mask.squeeze(0).squeeze(0).cpu()

            mask = self.pad_edge(mask, padding=20)
            mask_img = Image.fromarray(mask)
            output_image = self.inpainting(
                prompt=text2, image=image_pil, mask_image=mask_img)
            output_path = get_new_image_path(
                image_path, func_name='obj-replace')
            output_image = output_image.resize(image_pil.size)
            output_image.save(output_path)
            return output_path

    def pad_edge(self, mask, padding):
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        return new_mask

    def get_mask_with_boxes(self, image_pil, image, boxes_filt):
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]).to(self.device)

        features = self.sam_predictor.get_image_embedding(image)

        masks, _, _ = self.sam_predictor.predict_torch(
            features=features,
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks
