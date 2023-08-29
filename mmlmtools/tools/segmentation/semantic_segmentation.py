# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv

from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ...utils.file import get_new_image_path
from ..base_tool import BaseTool
from ..parsers import BaseParser


def load_semseg_inferencer(model, device):
    if CACHED_TOOLS.get('semseg_inferencer', None) is not None:
        semseg_inferencer = CACHED_TOOLS['semseg_inferencer'][model]
    else:
        try:
            from mmseg.apis import MMSegInferencer
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `mmseg` correctly')

        semseg_inferencer = MMSegInferencer(model=model, device=device)
        CACHED_TOOLS['semseg_inferencer'][model] = semseg_inferencer

    return semseg_inferencer


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
        self._inferencer = load_semseg_inferencer(self.toolmeta.model['model'],
                                                  self.device)

    def apply(self, inputs: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            results = self._inferencer(inputs, return_datasamples=True)
            output_path = get_new_image_path(
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
