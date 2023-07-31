# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Optional, Tuple

import cv2
import mmengine
import numpy as np
import torch
import wget
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..utils.file import get_new_image_path
from .base_tool import BaseTool
from .parsers import BaseParser
from .text2bbox import load_grounding

GLOBAL_SEED = 1912


def load_sam_and_predictor(model, model_ckpt_path, e_mode, device):
    if CACHED_TOOLS.get('sam', None) is not None:
        sam = CACHED_TOOLS['sam']
    else:
        url = ('https://dl.fbaipublicfiles.com/segment_anything/'
               f'{model}')
        mmengine.mkdir_or_exist('model_zoo')
        if not os.path.exists(model_ckpt_path):
            wget.download(url, out=model_ckpt_path)

        sam = sam_model_registry['vit_h'](checkpoint=f'model_zoo/{model}')
        if e_mode is not True:
            sam.to(device=device)
        CACHED_TOOLS['sam'] = sam

    if CACHED_TOOLS.get('sam_predictor', None) is not None:
        sam_predictor = CACHED_TOOLS['sam_predictor']
    else:
        sam_predictor = SamPredictor(sam)
        CACHED_TOOLS['sam_predictor'] = sam_predictor
    return sam, sam_predictor


class SamPredictor:

    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = 'RGB',
    ) -> None:
        """Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            'RGB',
            'BGR',
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(
            2, 0, 1).contiguous()[None, :, :, :]

        return self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:])
                == self.model.image_encoder.img_size), (
                    'set_torch_image input must be BCHW with long side'
                    f' {self.model.image_encoder.img_size}.')

        original_size = original_image_size
        input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        features = self.model.image_encoder(input_image)

        res = {
            'features': features,
            'original_size': original_size,
            'input_size': input_size
        }
        return res

    def predict(
        self,
        features,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks for the given input prompts, using the currently set
        image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to
            the model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model,
            typically coming from a previous prediction iteration. Has form
            1xHxW, where for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will
            often produce better masks than a single prediction. If only a
            single mask is needed, the model's predicted quality score can be
            used to select the best mask. For non-ambiguous prompts, such as
            multiple input prompts, multimask_output=False can give better
            results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if features.get('features', None) is None:
            raise RuntimeError('An image must be set with .set_image(...)'
                               ' before mask prediction.')

        # Transform input prompts
        coords_torch, labels_torch = None, None
        box_torch, mask_input_torch = None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), 'point_labels must be supplied if point_coords is supplied.'
            point_coords = self.transform.apply_coords(
                point_coords, features['original_size'])
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[
                None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, features['original_size'])
            box_torch = torch.as_tensor(
                box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            features,
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        features,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks for the given input prompts, using the currently set
        image. Input prompts are batched torch tensors and are expected to
        already be transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts
            to the model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model,
            typically coming from a previous prediction iteration. Has form
            Bx1xHxW, where for SAM, H=W=256. Masks returned by a previous
            iteration of the predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will
            often produce better masks than a single prediction. If only a
            single mask is needed, the model's predicted quality score can be
            used to select the best mask. For non-ambiguous prompts, such as
            multiple input prompts, multimask_output=False can give better
            results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if features.get('features', None) is None:
            raise RuntimeError('An image must be set with .set_image(...)'
                               ' before mask prediction.')

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=features['features'],
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks,
                                             features['input_size'],
                                             features['original_size'])

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self, image) -> torch.Tensor:
        return self.set_image(image)

    @property
    def device(self) -> torch.device:
        return self.model.device


class SegmentAnything(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Segment Anything On Image',
        model={'model': 'sam_vit_h_4b8939.pth'},
        description='This is a useful tool '
        'when you want to segment anything in the image,'
        'like: segment anything from this image. '
        'It takes an {{{input:image}}} as the input, and returns an '
        '{{{input:image}}} representing the segmented image. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.toolmeta.model['model'], self.model_ckpt_path, self.e_mode,
            self.device)
        self.model_ckpt_path = f"model_zoo/{self.toolmeta.model['model']}"
        self.e_mode = True

    def apply(self, image_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            annos = self.segment_anything(image_path)
            full_img, _ = self.show_annos(annos)
            seg_all_image_path = get_new_image_path(image_path, 'sam')
            full_img.save(seg_all_image_path, 'PNG')
            return seg_all_image_path

    def segment_anything(self, img_path):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # to device
        if self.e_mode:
            self.sam.to(device=self.device)

        mask_generator = SamAutomaticMaskGenerator(self.sam)
        annos = mask_generator.generate(img)

        # to cpu
        if self.e_mode:
            self.sam.to(device='cpu')
        return annos

    def segment_by_mask(self, mask, features):
        random.seed(GLOBAL_SEED)
        idxs = np.nonzero(mask)
        num_points = min(max(1, int(len(idxs[0]) * 0.01)), 16)
        sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
        new_mask = []
        for i in range(len(idxs)):
            new_mask.append(idxs[i][sampled_idx])
        points = np.array(new_mask).reshape(2, -1).transpose(1, 0)[:, ::-1]
        labels = np.array([1] * num_points)

        res_masks, scores, _ = self.sam_predictor.predict(
            features=features,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        return res_masks[np.argmax(scores), :, :]

    def get_detection_map(self, img_path):
        annos = self.segment_anything(img_path)
        _, detection_map = self.show_anns(annos)

        return detection_map

    def show_annos(self, anns):
        # From https://github.com/sail-sg/EditAnything/blob/main/sam2image.py#L91  # noqa
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        full_img = None

        # for ann in sorted_anns:
        for i in range(len(sorted_anns)):
            ann = anns[i]
            m = ann['segmentation']
            if full_img is None:
                full_img = np.zeros((m.shape[0], m.shape[1], 3))
                map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
            map[m != 0] = i + 1
            color_mask = np.random.random((1, 3)).tolist()[0]
            full_img[m != 0] = color_mask
        full_img = full_img * 255
        # anno encoding from https://github.com/LUSSeg/ImageNet-S
        res = np.zeros((map.shape[0], map.shape[1], 3))
        res[:, :, 0] = map % 256
        res[:, :, 1] = map // 256
        res.astype(np.float32)
        full_img = Image.fromarray(np.uint8(full_img))
        return full_img, res

    def get_image_embedding(self, img):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        # to device
        if self.e_mode:
            self.sam.to(device=self.device)

        embedding = self.sam_predictor.set_image(img)

        # to cpu
        if self.e_mode:
            self.sam.to(device='cpu')
        return embedding


class SegmentClicked(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Segment The Clicked Region In The Image',
        model={'model': 'sam_vit_h_4b8939.pth'},
        description='This is a useful tool '
        'when you want to segment the masked region or block in the image,'
        'like: segment the masked region in this image, '
        'The input to this tool should be an {{{input:image}}} representing '
        'the image, and an {{{input:image}}} representing the mask. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.toolmeta.model['model'], self.model_ckpt_path, self.e_mode,
            self.device)
        self.model_ckpt_path = f"model_zoo/{self.toolmeta.model['model']}"
        self.e_mode = True

    def apply(self, image_path, mask_path):
        if self.remote:
            raise NotImplementedError
        else:
            img = Image.open(image_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            features = self.get_image_embedding(img)

            clicked_mask = Image.open(mask_path).convert('L')
            clicked_mask = np.array(clicked_mask, dtype=np.uint8)

            res_mask = self.segment_by_mask(clicked_mask, features)

            res_mask = res_mask.astype(np.uint8) * 255
            filaname = get_new_image_path(image_path, 'sam-clicked')
            mask_img = Image.fromarray(res_mask)
            mask_img.save(filaname, 'PNG')

            return filaname

    def segment_by_mask(self, mask, features):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        # to device
        if self.e_mode:
            self.sam.to(device=self.device)

        random.seed(GLOBAL_SEED)
        idxs = np.nonzero(mask)
        num_points = min(max(1, int(len(idxs[0]) * 0.01)), 16)
        sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
        new_mask = []
        for i in range(len(idxs)):
            new_mask.append(idxs[i][sampled_idx])
        points = np.array(new_mask).reshape(2, -1).transpose(1, 0)[:, ::-1]
        labels = np.array([1] * num_points)

        res_masks, scores, _ = self.sam_predictor.predict(
            features=features,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # to cpu
        if self.e_mode:
            self.sam.to(device='cpu')

        return res_masks[np.argmax(scores), :, :]

    def get_image_embedding(self, img):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        # to device
        if self.e_mode:
            self.sam.to(device=self.device)

        embedding = self.sam_predictor.set_image(img)

        # to cpu
        if self.e_mode:
            self.sam.to(device='cpu')

        return embedding


class ObjectSegmenting(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Segment The Given Object In The Image',
        model={
            'model': 'sam_vit_h_4b8939.pth',
            'grounding': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'
        },
        description='This is a useful tool '
        'when you want to segment the certain objects in the image according '
        'to the given object name, like: segment the cat in this image, '
        'or can you segment an object for me. '
        'The input to this tool should be an {{{input:image}}} '
        'and a {{{input:text}}} representing the object description. '
        'It returns a {{{output:image}}} with the segmented objects. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.grounding = load_grounding(
            model=self.toolmeta.model['grounding'], device=self.device)

        self.e_mode = True
        self.model_ckpt_path = f"model_zoo/{self.toolmeta.model['model']}"
        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.toolmeta.model['model'], self.model_ckpt_path, self.e_mode,
            self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image_pil = Image.open(image_path).convert('RGB')

            results = self.grounding(
                inputs=image_path,
                texts=text,
                no_save_vis=True,
                return_datasample=True)
            results = results['predictions'][0].pred_instances

            boxes_filt = results.bboxes
            pred_phrases = results.label_names

            updated_image_path = self.segment_image_with_boxes(
                image_pil, image_path, boxes_filt, pred_phrases)
            return updated_image_path

    def get_mask_with_boxes(self, image_pil, image, boxes_filt):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

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

    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt,
                                 pred_phrases):
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image
        for mask in masks:
            image = self.show_mask(
                mask[0].cpu().numpy(),
                image,
                random_color=True,
                transparency=0.3)

        updated_image_path = get_new_image_path(
            image_path, func_name='segmentation')

        new_image = Image.fromarray(image)
        new_image.save(updated_image_path)

        return updated_image_path

    def show_mask(self,
                  mask: np.ndarray,
                  image: np.ndarray,
                  random_color: bool = False,
                  transparency=1) -> np.ndarray:
        """Visualize a mask on top of an image.

        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Outputs:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
            transparenccy: the transparency of the segmentation mask
        """

        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'),
                                transparency, 0)
        return image

    def show_box(self, box, ax, label):
        import matplotlib.pyplot as plt
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((x0, y0),
                          w,
                          h,
                          edgecolor='green',
                          facecolor=(0, 0, 0, 0),
                          lw=2))
        ax.text(x0, y0, label)
