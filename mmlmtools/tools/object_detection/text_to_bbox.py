# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


def load_grounding(model, device):
    if CACHED_TOOLS.get('grounding', None) is not None:
        grounding = CACHED_TOOLS['grounding'][model]
    else:
        try:
            from mmdet.apis import DetInferencer
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `mmdet` correctly')

        grounding = DetInferencer(model=model, device=device)
        CACHED_TOOLS['grounding'][model] = grounding
    return grounding


class TextToBbox(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Detect the Given Object',
        model={'model': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'},
        description='This is a useful tool when you only want to show the '
        'location of given objects, or detect or find out given objects in '
        'the picture. '
        'The input to this tool should be an {{{input:image}}} '
        'and a {{{input:text}}} representing the object description. '
        'It returns a {{{output:image}}} with the bounding box of the object.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_grounding(
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
            output_path = get_new_image_path(
                image, func_name='detect-something')
            img = mmcv.imread(image, channel_order='rgb')
            self._inferencer.visualizer.add_datasample(
                'results',
                img,
                data_sample=results['predictions'][0],
                show=False,
                out_file=output_path)
            return output_path
