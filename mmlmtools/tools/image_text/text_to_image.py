# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cache import load_or_build_object
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class TextToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image From User Input Text',
        model={'model': 'stable_diffusion'},
        description='This is a useful tool when you want to generate an image '
        'from a user input text and save it to a file.like: generate an image '
        'of an object or something, or generate an image that includes'
        'some objects. The input to this tool should be an {{{input:text}}} '
        'representing the object description. It returns a {{{output:image}}} '
        'representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

        self.aux_prompt = 'best quality, extremely detailed'

    def setup(self):
        from mmagic.apis import MMagicInferencer
        self._inferencer = load_or_build_object(
            MMagicInferencer,
            model_name=self.toolmeta.model['model'],
            model_setting=None,
            device=self.device)

    def apply(self, text: str) -> str:
        text += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_file_path(
                'image/sd-res.png', func_name='generate-image')
            self._inferencer.infer(text=text, result_out_dir=output_path)
            return output_path
