# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import mmcv
from mmdet.apis import DetInferencer

from mmlmtools.utils.toolmeta import ToolMeta
from ..cached_dict import CACHED_TOOLS
from ..utils.file import get_new_image_path
from .base_tool import BaseTool
from .parsers import BaseParser


def load_object_detection(model, device):
    if CACHED_TOOLS.get('object_detection', None) is not None:
        object_detection = CACHED_TOOLS['object_detection']
    else:
        object_detection = DetInferencer(model=model, device=device)
        CACHED_TOOLS['object_detection'] = object_detection

    return object_detection


class ObjectDetectionToolv2(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Detect All Objects',
        model={'model': 'rtmdet_l_8xb32-300e_coco'},
        description='This is a useful tool '
        'when you only want to detect the picture or detect all objects '
        'in the picture. like: detect all object or object. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_object_detection(
            model=self.toolmeta.model['model'], device=self.device)

    def apply(self, inputs):
        if self.remote:
            import json

            from openxlab.model import inference

            predict = inference('mmdetection/RTMDet', ['./demo_text_ocr.jpg'])
            print(f'json result:{json.loads(predict)}')
            raise NotImplementedError
        else:
            results = self._inferencer(
                inputs, no_save_vis=True, return_datasample=True)
            output_path = get_new_image_path(
                inputs, func_name='detect-something')
            img = mmcv.imread(inputs)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            self._inferencer.visualizer.add_datasample(
                'results',
                img,
                data_sample=results['predictions'][0],
                draw_gt=False,
                show=False,
                wait_time=0,
                out_file=output_path,
                pred_score_thr=0.5)
        return output_path