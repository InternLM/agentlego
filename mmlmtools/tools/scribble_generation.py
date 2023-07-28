# Copyright (c) OpenMMLab. All rights reserved.
from controlnet_aux import HEDdetector
from PIL import Image

from mmlmtools.toolmeta import ToolMeta
from ..utils.file import get_new_image_path
from .base_tool_v1 import BaseToolv1


class Image2ScribbleTool(BaseToolv1):
    DEFAULT_TOOLMETA = dict(
        name='Generate Scribble Conditioned On Image',
        model=None,
        description='This is a useful tool '
        'when you want to do the sketch detection on the image'
        'and generate the scribble. ',
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

        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    def setup(self):
        pass

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_path(
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
            scribble = self.detector(image, scribble=True)
            updated_image_path = get_new_image_path(
                inputs, func_name='scribble')
            scribble.save(updated_image_path)
        return updated_image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
