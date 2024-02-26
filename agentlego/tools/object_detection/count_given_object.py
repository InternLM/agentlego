from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class CountGivenObject(BaseTool):
    """A tool to count the number of a certain object in the image.

    Args:
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cpu'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'The tool can count the number of a certain object in the image.'

    @require('mmdet>=3.1.0')
    def __init__(self,
                 model: str = 'glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        from mmengine.registry import DefaultScope
        with DefaultScope.overwrite_default_scope('mmdet'):
            self._inferencer = load_or_build_object(
                DetInferencer, model=self.model, device=self.device)
        self._visualizer = self._inferencer.visualizer

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')]
    ) -> int:
        from mmdet.structures import DetDataSample

        results = self._inferencer(
            image.to_array()[:, :, ::-1],
            texts=text,
            return_datasamples=True,
        )
        data_sample: DetDataSample = results['predictions'][0]
        preds = data_sample.pred_instances

        if len(preds) == 0:
            return 0
        preds = preds[preds.scores > 0.5]

        return len(preds)
