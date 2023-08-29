# Copyright (c) OpenMMLab.All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


def load_mmagic_inferencer(model, setting, device):
    """Load mmagic inferencer.

    Args:
        model (str): The name of the model.
        setting (int): The setting of the model.
        device (str): The device to use.

    Returns:
        mmagic_inferencer (MMagicInferencer): The mmagic inferencer.
    """
    if CACHED_TOOLS.get('mmagic_inferencer' + str(setting), None) is not None:
        mmagic_inferencer = \
            CACHED_TOOLS['mmagic_inferencer' + str(setting)][model]
    else:
        try:
            from mmagic.apis import MMagicInferencer
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `mmagic` correctly')

        mmagic_inferencer = MMagicInferencer(
            model_name=model, model_setting=setting, device=device)
        CACHED_TOOLS['mmagic_inferencer' +
                     str(setting)][model] = mmagic_inferencer
    return mmagic_inferencer


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

    def setup(self):
        self.aux_prompt = 'best quality, extremely detailed'
        self._inferencer = load_mmagic_inferencer(self.toolmeta.model['model'],
                                                  None, self.device)

    def apply(self, text: str) -> str:
        text += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_file_path(
                'image/sd-res.png', func_name='generate-image')
            self._inferencer.infer(text=text, result_out_dir=output_path)
            return output_path
