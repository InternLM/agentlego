# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import numpy as np
from PIL import Image

from mmlmtools.parsers import DefaultParser
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from mmlmtools.utils.schema import ToolMeta
from ..base_tool import BaseTool


class OCR(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Recognize the Optical Characters On Image',
        description=('This is a useful tool when you want to recognize the '
                     'text from a photo. '),
        inputs=['image'],
        outputs=['text'],
    )

    @require(['mmocr>=1.0.1'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 det_model: str = 'dbnetpp',
                 rec_model: str = 'svtr-small',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.det_model = det_model
        self.rec_model = rec_model
        self.device = device

    def setup(self):
        from mmocr.apis import MMOCRInferencer
        self._inferencer = load_or_build_object(
            MMOCRInferencer,
            det=self.det_model,
            rec=self.rec_model,
            device=self.device)

    def apply(self, image: ImageIO) -> str:
        ocr_results = self._inferencer(image, show=False)['predictions'][0]
        outputs = ocr_results['rec_texts']
        return outputs


class ImageMaskOCR(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Recognize The Optical Characters On Image With Mask',
        description=('This is a useful tool when you want to recognize the '
                     'characters or words in the masked region of the image. '
                     'like: recognize the characters or words in the masked '
                     'region. '),
        inputs=['image', 'mask'],
        outputs=['text'],
    )

    @require(['mmocr>=1.0.1'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 det_model: str = 'dbnetpp',
                 rec_model: str = 'svtr-small',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.det_model = det_model
        self.rec_model = rec_model
        self.device = device

    def setup(self):
        from mmocr.apis import MMOCRInferencer
        self._inferencer = load_or_build_object(
            MMOCRInferencer,
            det=self.toolmeta.model['det'],
            rec=self.toolmeta.model['rec'],
            device=self.device)

    def apply(self, image: ImageIO, mask: ImageIO) -> str:
        image = image.strip()
        mask = mask.strip()
        mask = Image.open(mask).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        ocr_results = self._inferencer(image, show=False)
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

        # remove duplicates
        ocr_text_list = list(dict.fromkeys(ocr_text_list))
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
