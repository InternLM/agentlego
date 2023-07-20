# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
from mmdet.apis import DetInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Text2BoxTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Detect the Give Object',
        model={'model': 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365'},
        description='This is a useful tool '
        'when you only want to show the location of given objects, '
        'or detect or find out given objects in the picture.',
        input_description='The input to this tool should be '
        'a comma separated string of two, '
        'representing the image_path and the text description of objects. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            from mmlmtools.cached_dict import CACHED_TOOLS
            if CACHED_TOOLS.get('grounding', None) is not None:
                self._inferencer = CACHED_TOOLS['grounding']
            else:
                self._inferencer = DetInferencer(
                    model=self.toolmeta.model['model'], device=self.device)
                CACHED_TOOLS['grounding'] = self._inferencer

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, text = inputs
        if self.remote:
            raise NotImplementedError
        else:
            results = self._inferencer(
                inputs=image_path,
                texts=text,
                no_save_vis=True,
                return_datasample=True)
            output_path = get_new_image_name(
                image_path, func_name='detect-something')
            img = mmcv.imread(image_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            self._inferencer.visualizer.add_datasample(
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


class ObjectDetectionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Detect All Objects',
        model={'model': 'rtmdet_l_8xb32-300e_coco'},
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

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = DetInferencer(
                model=self.toolmeta.model['model'], device=self.device)

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

    def apply(self, inputs):
        if self.remote:
            import json

            from openxlab.model import inference

            predict = inference('mmdetection/RTMDet', ['./demo_text_ocr.jpg'])
            print(f'json result:{json.loads(predict)}')
            raise NotImplementedError
        else:
            results = self._inferencer(
                inputs, no_save_vis=True, return_datasample=True)
            output_path = get_new_image_name(
                inputs, func_name='detect-something')
            img = mmcv.imread(inputs)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            self._inferencer.visualizer.add_datasample(
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
