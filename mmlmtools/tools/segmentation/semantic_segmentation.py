# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv

from mmlmtools.parsers import BaseParser
from mmlmtools.schema import ToolMeta
from mmlmtools.utils.cache import load_or_build_object
from ...utils.file import get_new_file_path
from ..base import BaseTool


class SemanticSegmentation(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Segment the Image',
        model={'model': 'mask2former_r50_8xb2-90k_cityscapes-512x1024'},
        description='This is a useful tool '
        'when you only want to segment the picture or segment all '
        'objects in the picture. like: segment all objects. '
        'It takes an {{{input:image}}} as the input, and returns '
        'a {{{output:image}}} as the output, representing the image with '
        'semantic segmentation results. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        from mmseg.apis import MMSegInferencer
        self._inferencer = load_or_build_object(
            MMSegInferencer,
            model=self.toolmeta.model['model'],
            device=self.device)

    def apply(self, inputs: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            results = self._inferencer(inputs, return_datasamples=True)
            output_path = get_new_file_path(
                inputs, func_name='semseg-something')
            img = mmcv.imread(inputs)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            self._inferencer.visualizer.add_datasample(
                'results',
                img,
                data_sample=results,
                draw_gt=False,
                draw_pred=True,
                show=False,
                out_file=output_path)
        return output_path
