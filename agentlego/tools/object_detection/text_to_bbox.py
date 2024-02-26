from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class TextToBbox(BaseTool):
    """A tool to detection the given object.

    Args:
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cpu'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('The tool can detect the object location according to '
                    'description.')

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
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)
        self._visualizer = self._inferencer.visualizer

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')],
        top1: Annotated[bool,
                        Info('If true, return the object with highest score. '
                             'If false, return all detected objects.')] = True,
    ) -> Annotated[str,
                   Info('Detected objects, include bbox in '
                        '(x1, y1, x2, y2) format, and detection score.')]:
        from mmdet.structures import DetDataSample

        results = self._inferencer(
            image.to_array()[:, :, ::-1],
            texts=text,
            return_datasamples=True,
        )
        data_sample: DetDataSample = results['predictions'][0]
        preds = data_sample.pred_instances

        if len(preds) == 0:
            return 'No object found.'

        pred_tmpl = '({:.0f}, {:.0f}, {:.0f}, {:.0f}), score {:.0f}'
        if top1:
            preds = preds[preds.scores.topk(1).indices]
        else:
            preds = preds[preds.scores > 0.5]
        pred_descs = []
        for bbox, score in zip(preds.bboxes, preds.scores):
            pred_descs.append(pred_tmpl.format(*bbox, score * 100))
        pred_str = '\n'.join(pred_descs)

        return pred_str
