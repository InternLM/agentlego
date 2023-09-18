# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import cv2
import numpy as np
import torch
from PIL import Image

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import require
from mmlmtools.utils.cache import load_or_build_object
from ..base import BaseTool

GLOBAL_SEED = 1912


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
    """A tool to replace the certain objects in the image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        sam_model (str): The model name used to inference. Which can be found
            in the ``segment_anything`` repository.
            Defaults to ``sam_vit_h_4b8939.pth``.
        grounding_model (str): The model name used to inference. Which can be
            found in the ``MMdetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = dict(
        name='Replace Object In Image',
        description='This is a useful tool when you want to replace the '
        'certain objects in the image with another object, like replacing '
        'a cat in an image with a dog. You need to input the image to '
        'edit, the object name to be replaced, and the object to '
        'replace with.',
        inputs=['image', 'text', 'text'],
        outputs=['image'],
    )

    @require('mmdet')
    @require('segment_anything')
    @require('diffusers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 sam_model: str = 'sam_vit_h_4b8939.pth',
                 grounding_model:
                 str = 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser)
        self.sam_model = sam_model
        self.grounding_model = grounding_model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer

        from ..segmentation.segment_anything import load_sam_and_predictor

        self.grounding = load_or_build_object(
            DetInferencer, model=self.grounding_model, device=self.device)
        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.sam_model, False, self.device)

        self.inpainting = load_or_build_object(Inpainting, device=self.device)

    def apply(self, image: ImageIO, text1: str, text2: str) -> ImageIO:
        image_path = image.to_path()
        image_pil = image.to_pil()

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
        output_image = output_image.resize(image_pil.size)
        return ImageIO(output_image)

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
