from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class VQA(BaseTool):
    """A tool to answer the question about an image.

    Args:
        remote (bool): Whether to use the remote model. Defaults to False.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can answer the input question based on the '
                    'input image.')

    def __init__(self,
                 model: str = 'ofa-base_3rdparty-zeroshot_vqa',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta)
        self.device = device
        self.model = model

    @require('mmpretrain')
    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import VisualQuestionAnsweringInferencer

        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                VisualQuestionAnsweringInferencer, model=self.model, device=self.device)

    def apply(
        self,
        image: ImageIO,
        question: Annotated[str, Info('The question should be in English.')],
    ) -> str:
        image = image.to_array()[:, :, ::-1]
        return self._inferencer(image, question)[0]['pred_answer']
