# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from PIL import Image

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Image2CannyTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='Image2CannyTool',
        model='canny',
        description='This is a useful tool '
        'when you want to detect the edge of the image.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cpu',
                 low_threshold: int = 100,
                 high_threshold: int = 200):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def setup(self):
        pass

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
            raise NotImplementedError
        else:
            image = Image.open(inputs)
            image = np.array(image)
            canny = cv2.Canny(image, self.low_threshold,
                              self.high_threshold)[:, :, None]
            canny = np.concatenate([canny] * 3, axis=2)
            canny = Image.fromarray(canny)
            output_path = get_new_image_name(inputs, func_name='edge')
            canny.save(output_path)
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
