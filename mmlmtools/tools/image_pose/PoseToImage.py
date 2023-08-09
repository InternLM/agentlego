# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser

try:
    from mmagic.apis import MMagicInferencer
    has_mmagic = True
except ImportError:
    has_mmagic = False


def load_mmagic_inferencer(model, setting, device):
    if CACHED_TOOLS.get('mmagic_inferencer' + str(setting), None) is not None:
        mmagic_inferencer = \
            CACHED_TOOLS['mmagic_inferencer' + str(setting)][model]
    else:
        if not has_mmagic:
            raise RuntimeError('mmagic is required but not installed')
        mmagic_inferencer = MMagicInferencer(
            model_name=model, model_setting=setting, device=device)
        CACHED_TOOLS['mmagic_inferencer' +
                     str(setting)][model] = mmagic_inferencer
    return mmagic_inferencer


class PoseToImageTool(BaseTool):
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
        self._inferencer = load_mmagic_inferencer(
            self.toolmeta.model['model_name'],
            self.toolmeta.model['model_setting'], self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_path(
                'image/controlnet-res.png',
                func_name='generate-image-from-pose')
            self._inferencer.infer(
                text=text, control=image_path, result_out_dir=output_path)
        return output_path
