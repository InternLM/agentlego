# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
from mmseg.apis import MMSegInferencer
from mmengine import Registry

from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class SemSegTool(BaseTool):
    def __init__(self,
                 model: str = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024',
                 checkpoint: str = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.inferencer = MMSegInferencer(model, device=device, **kwargs)

    def convert_inputs(self, inputs, **kwargs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def infer(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmseg'):
                results = self.inferencer(inputs, return_datasamples=True)
                output_path = get_new_image_name(
                    inputs, func_name='semseg-something')
                img = mmcv.imread(inputs)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                self.inferencer.visualizer.add_datasample(
                    'results',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    draw_pred=True,
                    show=False,
                    out_file=output_path)
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
