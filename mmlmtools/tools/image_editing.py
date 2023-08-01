# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from mmlmtools.cached_dict import CACHED_TOOLS
from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import get_new_image_name

from .base_tool import BaseTool
from .parsers import BaseParser
from .segment_anything import load_sam_and_predictor

GLOBAL_SEED = 1912


class Inpainting:

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


class ObjectReplaceTool(BaseTool):
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
        from mmdet.apis import DetInferencer

        if CACHED_TOOLS.get('grounding', None) is not None:
            self.grounding = CACHED_TOOLS['grounding']
        else:
            self.grounding = DetInferencer(
                model=self.toolmeta.model['grounding'], device=self.device)
            CACHED_TOOLS['grounding'] = self.grounding

        if CACHED_TOOLS.get('sam', None) is not None and CACHED_TOOLS.get('predictor', None) is not None:  # noqa
            self.sam = CACHED_TOOLS['sam']
            self.sam_predictor = CACHED_TOOLS['sam_predictor']
        else:
            self.sam, self.sam_predictor = load_sam_and_predictor(
                self.toolmeta.model['model'],
                f"model_zoo/{self.toolmeta.model['model']}",
                True, self.device)
            CACHED_TOOLS['sam'] = self.sam
            CACHED_TOOLS['sam_predictor'] = self.sam_predictor

        if CACHED_TOOLS.get('inpainting', None) is not None:
            self.inpainting = CACHED_TOOLS['inpainting']
        else:
            self.inpainting = Inpainting(self.device)
            CACHED_TOOLS['inpainting'] = self.inpainting

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
                prompt=text2,
                image=image_pil,
                mask_image=mask_img)
            output_path = get_new_image_name(
                image, 'obj-replace')
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


class ObjectRemoveTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Remove the Given Object In The Image',
        model={
            'model': 'sam_vit_h_4b8939.pth',
            'grounding': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'
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
        from mmdet.apis import DetInferencer

        if CACHED_TOOLS.get('grounding', None) is not None:
            self.grounding = CACHED_TOOLS['grounding']
        else:
            self.grounding = DetInferencer(
                model=self.toolmeta.model['grounding'], device=self.device)
            CACHED_TOOLS['grounding'] = self.grounding

        if CACHED_TOOLS.get('sam', None) is not None and CACHED_TOOLS.get('predictor', None) is not None:  # noqa
            self.sam = CACHED_TOOLS['sam']
            self.sam_predictor = CACHED_TOOLS['sam_predictor']
        else:
            self.sam, self.sam_predictor = load_sam_and_predictor(
                self.toolmeta.model['model'],
                f"model_zoo/{self.toolmeta.model['model']}",
                True, self.device)
            CACHED_TOOLS['sam'] = self.sam
            CACHED_TOOLS['sam_predictor'] = self.sam_predictor

        if CACHED_TOOLS.get('inpainting', None) is not None:
            self.inpainting = CACHED_TOOLS['inpainting']
        else:
            self.inpainting = Inpainting(self.device)
            CACHED_TOOLS['inpainting'] = self.inpainting

        if CACHED_TOOLS.get('vqa', None) is not None:
            self.vqa = CACHED_TOOLS['vqa']
        else:
            from .vqa import VisualQuestionAnsweringTool
            self.vqa = VisualQuestionAnsweringTool(self.device)
            self.vqa.setup()
            CACHED_TOOLS['vqa'] = self.vqa

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image_pil = Image.open(image_path).convert('RGB')
            text1 = text
            text2 = self.vqa.apply(
                image_path, 'tell me what is the background of this image')
            results = self.grounding(
                inputs=image_path,
                texts=[text1],
                no_save_vis=True,
                return_datasample=True)
            results = results['predictions'][0].pred_instances

            boxes_filt = results.bboxes

            if self.sam is None or self.sam_predictor is None:
                self.setup()
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
                prompt=text2,
                image=image_pil,
                mask_image=mask_image)
            output_path = get_new_image_name(
                image, 'obj-remove')
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
