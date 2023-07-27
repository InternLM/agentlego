# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv
from .base_tool_v2 import BaseToolv2
from mmdet.apis import DetInferencer

from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import get_new_image_name
from .parsers import BaseParser


class Text2BboxToolv2(BaseToolv2):
    DEFAULT_TOOLMETA = dict(
        name='Detect the Given Object',
        model={'model': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'},
        description='This is a useful tool when you only want to show the '
        'location of given objects, or detect or find out given objects in '
        'the picture. The input to this tool should be an {{{input:image}}} '
        'and a {{{input:text}}} representing the object description. '
        'It returns a {{{output:image}}} with the bounding box of the object.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = DetInferencer(
            model=self.toolmeta.model['model'], device=self.device)

    def apply(self, image: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            results = self._inferencer(
                inputs=image,
                texts=text,
                no_save_vis=True,
                return_datasample=True)
            output_path = get_new_image_name(
                image, func_name='detect-something')
            img = mmcv.imread(image, channel_order='rgb')
            self._inferencer.visualizer.add_datasample(
                'results',
                img,
                results,
                show=False,
                out_file=output_path,
                texts=text)
            return output_path
