# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser
from ..segment_anything import load_sam_and_predictor
from .replace import load_grounding, load_inpainting

GLOBAL_SEED = 1912


class ObjectRemove(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Remove the Given Object In The Image',
        model={
            'model': 'sam_vit_h_4b8939.pth',
            'grounding': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
            'vqa': 'ofa-base_3rdparty-zeroshot_vqa'
        },
        description='This is a useful tool when you want to remove the '
        'certain objects in the image. like: remove the cat in this image'
        'The input to this tool should be an {{{input:image}}} and '
        '{{{input:text}}} representing the object to be removed.It returns '
        'an {{{output:image}}} representing the image with the removed object')

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

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image_pil = Image.open(image_path).convert('RGB')
            text1 = text
            text2 = 'background'
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
            mask_image = Image.fromarray(mask)
            output_image = self.inpainting(
                prompt=text2, image=image_pil, mask_image=mask_image)
            output_path = get_new_image_path(
                image_path, func_name='obj-remove')
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
