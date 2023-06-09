# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial

import mmcv
from mmdet.apis import DetInferencer, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine import Registry

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Text2BoxTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='Text2BoxTool',
        model='glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
        description='This is a useful tool '
        'when you only want to detect or find out '
        'given objects in the picture.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.model = DetInferencer(
                self.toolmeta.model, device=self.device).model
            self.inferencer = partial(inference_detector, model=self.model)
            self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs, **kwargs):
        image_path, text = inputs
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmdet'):
                results = self.inferencer(
                    imgs=image_path, text_prompt=text, **kwargs)
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

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class ObjectDetectionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='ObjectDetectionTool',
        model='rtmdet_l_8xb32-300e_coco',
        description='This is a useful tool '
        'when you only want to detect the picture or detect all objects '
        'in the picture. like: detect all object or object. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.inferencer = DetInferencer(
                self.toolmeta.model, device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmdet'):
                results = self.inferencer(
                    inputs, no_save_vis=True, return_datasample=True, **kwargs)
                output_path = get_new_image_name(
                    inputs, func_name='detect-something')
                img = mmcv.imread(inputs)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                self.inferencer.visualizer.add_datasample(
                    'results',
                    img,
                    data_sample=results['predictions'][0],
                    draw_gt=False,
                    show=False,
                    wait_time=0,
                    out_file=output_path,
                    pred_score_thr=0.5)
        return output_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
