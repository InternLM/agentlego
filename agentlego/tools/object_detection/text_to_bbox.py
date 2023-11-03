# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Tuple, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class TextToBbox(BaseTool):
    """A tool to detection the given object.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='Detect the Given Object',
        description='The tool can detect the object location according to '
        'description in English. It will return an image with a bbox of the '
        'detected object, and the coordinates of bbox.',
        inputs=['image', 'text'],
        outputs=['image', 'text'],
    )

    @require('mmdet>=3.1.0')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)
        self._visualizer = self._inferencer.visualizer

    def apply(self, image: ImageIO, text: str) -> Tuple[ImageIO, str]:
        from mmdet.structures import DetDataSample

        results = self._inferencer(
            image.to_array()[:, :, ::-1],
            texts=text,
            return_datasamples=True,
        )
        data_sample = results['predictions'][0]
        preds: DetDataSample = data_sample.pred_instances

        if len(preds) > 0:
            topk = preds.scores.topk(1).indices
            preds = preds[topk]
            score = preds.scores[0].item()
            bbox = preds.bboxes[0].tolist()
            pred_str = ('bbox: ({:.0f}, {:.0f}, {:.0f}, {:.0f}), '
                        'score: {:.2f} (max 1.0)')
            pred_str = pred_str.format(*bbox, score)

            data_sample.pred_instances = preds
            self._visualizer.add_datasample(
                'vis', image.to_array(), data_sample, draw_gt=False)
            output_image = ImageIO(self._visualizer.get_image())
        else:
            pred_str = 'No object found.'
            output_image = image

        return output_image, pred_str
