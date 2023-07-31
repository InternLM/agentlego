# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmpretrain.apis import ImageCaptionInferencer

from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from .base_tool import BaseTool
from .parsers import BaseParser


def load_caption_inferencer(model, device):
    if CACHED_TOOLS.get('caption_inferencer', None) is not None:
        caption_inferencer = CACHED_TOOLS['caption_inferencer'][model]
    else:
        caption_inferencer = ImageCaptionInferencer(model=model, device=device)
        CACHED_TOOLS['caption_inferencer'][model] = caption_inferencer

    return caption_inferencer


class ImageCaptionTool(BaseTool):

    DEFAULT_TOOLMETA = dict(
        name='Get Photo Description',
        model={'model': 'blip-base_3rdparty_caption'},
        description='This is a useful tool when you want to know '
        'what is inside the image. It takes an {{{input:image}}} as the '
        'input, and returns a {{{output:text}}} representing the description '
        'of the image. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cpu'):

        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_caption_inferencer(
            self.toolmeta.model['model'], self.device)

    def apply(self, image: np.ndarray) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            return self._inferencer(image)[0]['pred_caption']
