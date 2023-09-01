# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from PIL import Image

from mmlmtools.utils.cache import load_or_build_object
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class OCR(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Recognize the Optical Characters On Image',
        model={
            'det': 'dbnetpp',
            'rec': 'svtr-small'
        },
        description='This is a useful tool when you want to '
        'recognize the text from a photo. It takes an {{{input:image}}} as '
        'the input, and returns a {{{output:text}}} as the output, '
        'representing the characters or words in the image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        from mmocr.apis import MMOCRInferencer
        self._inferencer = load_or_build_object(
            MMOCRInferencer,
            det=self.toolmeta.model['det'],
            rec=self.toolmeta.model['rec'],
            device=self.device)

    def apply(self, image: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            ocr_results = self._inferencer(image, show=False)['predictions'][0]
            outputs = ocr_results['rec_texts']
        return outputs


class ImageMaskOCR(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Recognize The Optical Characters On Image With Mask',
        model={
            'det': 'dbnetpp',
            'rec': 'svtr-small'
        },
        description='This is a useful tool when you want to '
        'recognize the characters or words in the masked '
        'region of the image. '
        'like: recognize the characters or words in the masked region. '
        'The input to this tool should be an {{{input:image}}} representing '
        'the image, and an {{{input:image}}} representing the mask. '
        'It returns a {{{output:text}}} representing the characters or words '
        'in the masked region. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        from mmocr.apis import MMOCRInferencer
        self._inferencer = load_or_build_object(
            MMOCRInferencer,
            det=self.toolmeta.model['det'],
            rec=self.toolmeta.model['rec'],
            device=self.device)

    def apply(self, image_path: str, mask_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image_path = image_path.strip()
            mask_path = mask_path.strip()
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.uint8)
            ocr_results = self._inferencer(image_path, show=False)
            ocr_results = ocr_results['predictions'][0]
            seleted_ocr_text = self.get_ocr_by_mask(mask, ocr_results)

        return seleted_ocr_text

    def get_ocr_by_mask(self, mask, ocr_res):
        inds = np.where(mask != 0)
        inds = (inds[0][::8], inds[1][::8])

        if len(inds[0]) == 0:
            return 'No characters in the image'

        ocr_text_list = []
        num_mask = len(inds[0])
        for i in range(num_mask):
            res = self.search((inds[1][i], inds[0][i]), ocr_res)
            if res is not None and len(res) > 0:
                ocr_text_list.append(res)
        ocr_text_list = list(dict.fromkeys(ocr_text_list))  # remove duplicates
        ocr_text = '\n'.join(ocr_text_list)
        if ocr_text is None or len(ocr_text.strip()) == 0:
            ocr_text = 'No characters in the image'
        else:
            ocr_text = '\n' + ocr_text

        return ocr_text

    def search(self, coord, ocr_res):
        det_bboxes = ocr_res.get('det_bboxes', [])
        rec_texts = ocr_res.get('rec_texts', [])
        for i, item in det_bboxes:
            left_top = item[:2]
            right_bottom = item[2:]
            if (coord[0] >= left_top[0] and coord[1] >= left_top[1]) and \
               (coord[0] <= right_bottom[0] and coord[1] <= right_bottom[1]):
                return rec_texts[i]

        return ''
