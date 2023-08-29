# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from PIL import Image

from mmlmtools.utils.file import get_new_image_path
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class ImageToDepth(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Depth Image On Image',
        model=None,
        description='This is a useful tool when you want to'
        'generate the depth image of an image. It takes an {{{input:image}}} '
        'as the input, and returns a {{{output:image}}} representing the '
        'depth image of the input image. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):

        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `transformers` correctly')

        self.depth_estimator = pipeline('depth-estimation')

    def apply(self, image_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(image_path)
            depth = self.depth_estimator(image)['depth']
            depth = np.array(depth)
            depth = depth[:, :, None]
            depth = np.concatenate([depth, depth, depth], axis=2)
            depth = Image.fromarray(depth)
            output_path = get_new_image_path(image_path, func_name='depth')
            depth.save(output_path)
            return output_path
