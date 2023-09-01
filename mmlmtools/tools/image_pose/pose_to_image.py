# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cache import load_or_build_object
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class PoseToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Pose Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 2
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a human pose image and the user description. like: '
        'generate a real image of a human from this human pose image. or '
        'generate a real image of a human from this pose. The input to this '
        'tool should be an {{{input:image}}} and a {{{input:text}}} '
        'representing the image and the text description. It returns a '
        '{{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        from mmagic.apis import MMagicInferencer
        self._inferencer = load_or_build_object(
            MMagicInferencer,
            model_name=self.toolmeta.model['model_name'],
            model_setting=self.toolmeta.model['model_setting'],
            device=self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_file_path(
                'image/controlnet-res.png',
                func_name='generate-image-from-pose')
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path
