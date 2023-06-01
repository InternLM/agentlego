# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial

import mmcv
from mmdet.apis import DetInferencer, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine import Registry

from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Text2BoxTool(BaseTool):

    def __init__(self,
                 model: str = '/home/PJLAB/jiangtao/Documents/git-clone/'
                 'mmdetection/configs/glip/'
                 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py',
                 checkpoint: str = None,
                 input_style: str = 'image_path, text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.model = DetInferencer(
            'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
            device=device).model
        self.inferencer = partial(inference_detector, model=self.model)
        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)

    def convert_inputs(self, inputs, **kwargs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def infer(self, inputs, **kwargs):
        image_path, text = inputs
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmdet'):
                results = self.inferencer(imgs=image_path, text_prompt=text)
                output_path = get_new_image_name(
                    image_path, func_name='detect-something')
                img = mmcv.imread(image_path)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                self.visualizer.add_datasample(
                    'results',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    show=False,
                    wait_time=0,
                    out_file=output_path,
                    pred_score_thr=0.5)

        return output_path

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
