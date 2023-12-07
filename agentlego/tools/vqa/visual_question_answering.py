from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class VisualQuestionAnswering(BaseTool):
    """A tool to answer the question about an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        remote (bool): Whether to use the remote model. Defaults to False.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='VQA',
        description='This tool can answer the input question based on the '
        'input image. The question should be in English.',
        inputs=['image', 'text'],
        outputs=['text'])

    def __init__(self,
                 toolmeta: Union[ToolMeta, dict] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'ofa-base_3rdparty-zeroshot_vqa',
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser)
        self.device = device
        self.model = model

    @require('mmpretrain')
    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import VisualQuestionAnsweringInferencer

        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                VisualQuestionAnsweringInferencer,
                model=self.model,
                device=self.device)

    def apply(self, image: ImageIO, text: str) -> str:
        image = image.to_array()[:, :, ::-1]
        return self._inferencer(image, text)[0]['pred_answer']
