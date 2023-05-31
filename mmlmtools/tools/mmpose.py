# Copyright (c) OpenMMLab. All rights reserved.
import shutil

from mmengine import Registry
from mmpose.apis import MMPoseInferencer
from PIL import Image

from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class HumanBodyPoseTool(BaseTool):

    def __init__(self,
                 model: str = 'human',
                 checkpoint: str = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.inferencer = MMPoseInferencer(model, device=device, **kwargs)

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
            with Registry('scope').switch_scope_and_registry('mmpose'):
                next(self.inferencer(inputs, vis_out_dir='image/pose-res/'))
            src_path = 'image/pose-res/' + inputs.split('/')[-1]
            image_path = get_new_image_name(
                'image/' + inputs.split('/')[-1], func_name='pose-estimation')
            shutil.move(src_path, image_path)
        return image_path

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'image':   # transformer agent style
            img = Image.open(outputs)
            return img

        else:
            raise NotImplementedError
