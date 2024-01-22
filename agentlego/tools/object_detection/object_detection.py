from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class ObjectDetection(BaseTool):
    """A tool to detection all objects defined in COCO 80 classes.

    Args:
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``rtmdet_l_8xb32-300e_coco``.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'The tool can detect all common objects in the picture.'

    @require('mmdet>=3.1.0')
    def __init__(self,
                 model: str = 'rtmdet_l_8xb32-300e_coco',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)
        self.classes = self._inferencer.model.dataset_meta['classes']

    def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('All detected objects, include object name, '
                        'bbox in (x1, y1, x2, y2) format, '
                        'and detection score.')]:
        from mmdet.structures import DetDataSample

        results = self._inferencer(
            image.to_array()[:, :, ::-1],
            return_datasamples=True,
        )
        data_sample = results['predictions'][0]
        preds: DetDataSample = data_sample.pred_instances
        preds = preds[preds.scores > 0.5]
        pred_descs = []
        pred_tmpl = '{} ({:.0f}, {:.0f}, {:.0f}, {:.0f}), score {:.0f}'
        for label, bbox, score in zip(preds.labels, preds.bboxes, preds.scores):
            label = self.classes[label]
            pred_descs.append(pred_tmpl.format(label, *bbox, score * 100))
        if len(pred_descs) == 0:
            return 'No object found.'
        else:
            return '\n'.join(pred_descs)
