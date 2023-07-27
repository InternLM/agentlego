# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from PIL import Image
from transformers import pipeline

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Image2DepthTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Depth Image On Image',
        model=None,
        description='This is a useful tool '
        'when you want to generate the depth image of an image. ',
        input_description='It takes a string as the input, '
        'representing the image_path. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)
        self.depth_estimator = pipeline('depth-estimation')

    def setup(self):
        pass

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':
            return inputs
        elif self.input_style == 'pil image':
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            image = Image.open(inputs)
            depth = self.depth_estimator(image)['depth']
            depth = np.array(depth)
            depth = depth[:, :, None]
            depth = np.concatenate([depth, depth, depth], axis=2)
            depth = Image.fromarray(depth)
            output_path = get_new_image_name(inputs, func_name='depth')
            depth.save(output_path)
            return output_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
