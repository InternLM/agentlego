# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmpretrain.apis import VisualQuestionAnsweringInferencer
from mmlmtools.toolmeta import ToolMeta

from .base_tool import BaseTool
from .parsers import BaseParser


class VisualQuestionAnsweringTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Visual Question Answering',
        model={'model': 'ofa-base_3rdparty-zeroshot_vqa'},
        description='This is a useful tool '
        'when you want to know some information about the image.'
        'you can ask questions like "what is the color of the car?"'
        'The input to this tool should be an {{{input:image}}} and a '
        '{{{input:text}}}, representing the image and the question.'
        'It returns a {{{output:text}}} as the output, representing '
        'the answer to the question. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = VisualQuestionAnsweringInferencer(
            model=self.toolmeta.model['model'], device=self.device)

    def apply(self, image_path: str, text: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            return self._inferencer(image_path, text)[0]['pred_answer']
