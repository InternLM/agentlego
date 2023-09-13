# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import mmcv

from mmlmtools.parsers import BaseParser
from mmlmtools.schema import ToolMeta
from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cache import load_or_build_object
from ..base import BaseTool


class ObjectDetection(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Detect All Objects',
        model={'model': 'rtmdet_l_8xb32-300e_coco'},
        description='This is a useful tool '
        'when you only want to detect the picture or detect all objects '
        'in the picture. like: detect all object or object. '
        'It takes an {{{input:image}}} as the input, and returns '
        'a {{{output:image}}} as the output, representing the image with '
        'bounding boxes of all objects. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        if self.remote:
            try:
                from openxlab.model import inference as openxlab_inference
            except ImportError as e:
                raise ImportError(
                    f'Failed to run the tool for {e}, please check if you have'
                    ' install `openxlab` correctly')

            self._inferencer = openxlab_inference
        else:
            from mmdet.apis import DetInferencer
            self._inferencer = load_or_build_object(
                DetInferencer,
                model=self.toolmeta.model['model'],
                device=self.device)

    def apply(self, image_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            results = self._inferencer(
                image_path, no_save_vis=True, return_datasample=True)
            output_path = get_new_file_path(
                image_path, func_name='detect-something')
            img = mmcv.imread(image_path)
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
