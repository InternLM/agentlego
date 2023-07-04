# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.apis import MMagicInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class Text2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image From User Input Text',
        model={'model_name': 'stable_diffusion'},
        description='This is a useful tool '
        'when you want to generate an image from'
        'a user input text and save it to a file. like: generate '
        'an image of an object or something, or generate an image '
        'that includes some objects.',
        input_description='It takes a string as the input, '
        'representing the text that the tool required. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self.aux_prompt = 'best quality, extremely detailed'
            self._inferencer = MMagicInferencer(
                model_name=self.toolmeta.model['model_name'],
                device=self.device)

    def apply(self, inputs):
        inputs += self.aux_prompt
        if self.remote:
            raise NotImplementedError
        else:
            image_path = get_new_image_name(
                'image/sd-res.png', func_name='generate-image')
            self._inferencer.infer(text=inputs, result_out_dir=image_path)
        return image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class Seg2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Segmentations',
        model={
            'model_name': 'controlnet',
            'model_setting': 3
        },
        description='This is a useful tool '
        'when you want to generate a new real image from a segmentation image and '  # noqa
        'the user description. like: generate a real image of a '
        'object or something from this segmentation image. or generate a '
        'new real image of a object or something from this segmentation image. ',  # noqa
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image_path of a segmentation '
        'image and the text description of objects to generate.')

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
            self._inferencer = MMagicInferencer(
                model_name=self.toolmeta.model['model_name'],
                model_setting=self.toolmeta.model['model_setting'],
                device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, prompt = inputs
        if self.remote:
            raise NotImplementedError
        else:
            out_path = get_new_image_name(
                'image/controlnet-res.png',
                func_name='generate-image-from-seg')
            self._inferencer.infer(
                text=prompt, control=image_path, result_out_dir=out_path)
        return out_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class Canny2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Canny Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 1
        },
        description='This is a useful tool '
        'when you want to generate a new real image from a canny image and '
        'the user description. like: generate a real image of a '
        'object or something from this canny image. or generate a '
        'new real image of a object or something from this edge image. ',
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image_path of a canny '
        'image and the text description of objects to generate.')

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
            self._inferencer = MMagicInferencer(
                model_name=self.toolmeta.model['model_name'],
                model_setting=self.toolmeta.model['model_setting'],
                device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, prompt = inputs
        if self.remote:
            raise NotImplementedError
        else:
            out_path = get_new_image_name(
                'image/controlnet-res.png',
                func_name='generate-image-from-canny')
            self._inferencer.infer(
                text=prompt, control=image_path, result_out_dir=out_path)
        return out_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class Pose2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image Condition On Pose Image',
        model={
            'model_name': 'controlnet',
            'model_setting': 2
        },
        description='This is a useful tool '
        'when you want to generate a new real image from a human pose image and '
        'the user description. like: generate a real image of a human from this human pose image. '
        'or generate a new real image of a human from this pose. ',
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image_path of a human pose '
        'image and the text description of objects to generate.')

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
            self._inferencer = MMagicInferencer(
                model_name=self.toolmeta.model['model_name'],
                model_setting=self.toolmeta.model['model_setting'],
                device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            splited_inputs = inputs.split(',')
            image_path = splited_inputs[0]
            text = ','.join(splited_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, prompt = inputs
        if self.remote:
            raise NotImplementedError
        else:
            out_path = get_new_image_name(
                'image/controlnet-res.png',
                func_name='generate-image-from-pose')
            self._inferencer.infer(
                text=prompt, control=image_path, result_out_dir=out_path)
        return out_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
