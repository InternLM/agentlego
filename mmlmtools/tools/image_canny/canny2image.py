# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

from mmagic.apis import MMagicInferencer

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


def load_mmagic_inferencer(model, setting, device):
    if CACHED_TOOLS.get('mmagic_inferencer' + str(setting), None) is not None:
        mmagic_inferencer = \
            CACHED_TOOLS['mmagic_inferencer' + str(setting)][model]
    else:
        mmagic_inferencer = MMagicInferencer(
            model_name=model, model_setting=setting, device=device)
        CACHED_TOOLS['mmagic_inferencer' +
                     str(setting)][model] = mmagic_inferencer
    return mmagic_inferencer


class Canny2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Canny Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 1
        },
        description='This is a useful tool when you want to generate a new '
        'real image from a canny image and the user description. like: '
        'generate a real image of a object or something from this canny image.'
        'The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}} representing the image and the text description. '
        'It returns a {{{output:image}}} representing the generated image.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_mmagic_inferencer(
            self.toolmeta.model['model_name'],
            self.toolmeta.model['model_setting'], self.device)

    def apply(self, image_path: str, text: str) -> str:
        output_path = get_new_image_path(
            'image/controlnet-res.png', func_name='generate-image-from-canny')

        if self.remote:
            from openxlab.model import inference
            out = inference('mmagic/controlnet_canny', [image_path, text])
            with open(output_path, 'wb') as file:
                file.write(out)

        else:
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path
