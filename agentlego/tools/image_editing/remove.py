import numpy as np
from PIL import Image

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool

GLOBAL_SEED = 1912


class ObjectRemove(BaseTool):
    """A tool to remove the certain objects in the image.

    Args:
        sam_model (str): The model name used to inference. Which can be found
            in the ``segment_anything`` repository.
            Defaults to ``sam_vit_h_4b8939.pth``.
        grounding_model (str): The model name used to inference. Which can be
            found in the ``MMdetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can remove the specified object in the image.'

    @require('mmdet')
    @require('segment_anything')
    @require('diffusers')
    def __init__(self,
                 sam_model: str = 'sam_vit_h_4b8939.pth',
                 grounding_model: str = 'glip_atss_swin-t_a'
                 '_fpn_dyhead_pretrain_obj365',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta)
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer

        from ..segmentation.segment_anything import load_sam_and_predictor
        from .replace import Inpainting

        self.grounding = load_or_build_object(
            DetInferencer, model=self.grounding_model, device=self.device)
        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.sam_model, device=self.device)

        self.inpainting = load_or_build_object(Inpainting, device=self.device)

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object to remove.')],
    ) -> ImageIO:
        import torch
        image_path = image.to_path()
        image_pil = image.to_pil()

        text1 = text
        text2 = 'background'
        results = self.grounding(
            inputs=image_path, texts=[text1], no_save_vis=True, return_datasamples=True)
        results = results['predictions'][0].pred_instances

        boxes_filt = results.bboxes

        self.sam_predictor.set_image(image.to_array())
        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()

        mask = self.pad_edge(mask, padding=20)
        mask_image = Image.fromarray(mask)
        output_image = self.inpainting(
            prompt=text2, image=image_pil, mask_image=mask_image)
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
