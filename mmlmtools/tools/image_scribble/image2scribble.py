# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from controlnet_aux import HEDdetector
from PIL import Image

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class Image2ScribbleTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Scribble Conditioned On Image',
        model=None,
        description='This is a useful tool when you want to do '
        'the sketch detection on the image and generate the scribble. '
        'It takes an {{{input:image}}} as the input, and returns a '
        '{{{output:image}}} representing the scribble of the image. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):

        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    def apply(self, image_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(image_path)
            scribble = self.detector(image, scribble=True)
            output_path = get_new_image_path(image_path, func_name='scribble')
            scribble.save(output_path)
        return output_path
