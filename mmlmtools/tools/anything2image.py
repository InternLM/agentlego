# Copyright (c) OpenMMLab. All rights reserved.
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from . import imagebind_huge as ib
from .base_tool import BaseTool


class Anything2Image:

    def __init__(self, device, e_mode):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-unclip',
            torch_dtype=torch.float16,
            variation='fp16')
        self.device = device
        self.e_mode = e_mode
        self.pipe = pipe
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.model = ib.imagebind_huge(pretrained=True)
        self.model.eval()
        if self.e_mode is not True:
            self.pipe.to(device)
            self.model.to(device)


class Audio2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Audio',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from audio. '
        'like: generate a real image from audio, '
        'or generate a new image based on the given audio. ',
        input_description='It takes a string as the input, '
        'representing the audio_path. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'audio_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = Anything2Image(self.device, True)
            self.pipe = self._inferencer.pipe
            self.model = self._inferencer.model
            self.device = self._inferencer.device
            self.e_mode = self._inferencer.e_mode

    def convert_inputs(self, inputs):
        if self.input_style == 'audio_path':  # visual chatgpt style
            return inputs
        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            if self.e_mode:
                self.pipe.to(self.device)
                self.model.to(self.device)

            audio_paths = [inputs]
            audio_data = ib.load_and_transform_audio_data(
                audio_paths, self.device)
            embeddings = self.model.forward(
                {ib.ModalityType.AUDIO: audio_data})
            embeddings = embeddings[ib.ModalityType.AUDIO]
            images = self.pipe(
                image_embeds=embeddings.half(), width=512, height=512).images
            new_img_name = get_new_image_name(audio_paths[0], 'Audio2Image')
            images[0].save(new_img_name)

            if self.e_mode:
                self.pipe.to('cpu')
                self.model.to('cpu')

        return new_img_name

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class Thermal2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Thermal Image',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from a thermal image. '
        'like: generate a real image from thermal image, '
        'or generate a new image based on the given thermal image. ',
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
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = Anything2Image(self.device, True)
            self.pipe = self._inferencer.pipe
            self.model = self._inferencer.model
            self.device = self._inferencer.device
            self.e_mode = self._inferencer.e_mode

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
            if self.e_mode:
                self.pipe.to(self.device)
                self.model.to(self.device)

            thermal_paths = [inputs]
            thermal_data = ib.load_and_transform_thermal_data(
                thermal_paths, self.device)
            embeddings = self.model.forward(
                {ib.ModalityType.THERMAL: thermal_data})
            embeddings = embeddings[ib.ModalityType.THERMAL]
            images = self.pipe(
                image_embeds=embeddings.half(), width=512, height=512).images
            new_img_name = get_new_image_name(thermal_data[0], 'Thermal2Image')
            images[0].save(new_img_name)

            if self.e_mode:
                self.pipe.to('cpu')
                self.model.to('cpu')

        return new_img_name

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class AudioImage2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Image and Audio',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from image and audio. '
        'like: generate a real image from image and audio, '
        'or generate a new image based on the given image and audio. ',
        input_description='The input to this tool should be a comma separated'
        ' string of two, representing the image_path and audio_path. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, audio_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = Anything2Image(self.device, True)
            self.pipe = self._inferencer.pipe
            self.model = self._inferencer.model
            self.device = self._inferencer.device
            self.e_mode = self._inferencer.e_mode

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, audio_path':  # visual chatgpt style  # noqa
            image_path, audio_path = inputs.split(',')
            image_path, audio_path = image_path.strip(), audio_path.strip()
            return image_path, audio_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        image_path, audio_path = inputs
        if self.remote:
            raise NotImplementedError
        else:
            if self.e_mode:
                self.pipe.to(self.device)
                self.model.to(self.device)

            print(f'AudioImage2Image: {inputs}')

            # process image data
            vision_data = ib.load_and_transform_vision_data([image_path],
                                                            self.device)
            embeddings = self.model.forward(
                {
                    ib.ModalityType.VISION: vision_data,
                }, normalize=False)
            img_embeddings = embeddings[ib.ModalityType.VISION]

            # process audio data
            audio_data = ib.load_and_transform_audio_data([audio_path],
                                                          self.device)
            embeddings = self.model.forward({
                ib.ModalityType.AUDIO: audio_data,
            })
            audio_embeddings = embeddings[ib.ModalityType.AUDIO]

            embeddings = (img_embeddings + audio_embeddings) / 2
            images = self.pipe(
                image_embeds=embeddings.half(), width=512, height=512).images
            new_img_name = get_new_image_name(audio_path, 'AudioImage2Image')
            images[0].save(new_img_name)

            if self.e_mode:
                self.pipe.to('cpu')
                self.model.to('cpu')

        return new_img_name

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class AudioText2ImageTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Audio and Text',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from audio and text prompt. '
        "like: generate a real image from audio with user's prompt, "
        'or generate a new image based on the given image audio with '
        "user's description. ",
        input_description='The input to this tool should be a comma separated'
        ' string of two, representing the audio_path and prompt. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'audio_path, text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = Anything2Image(self.device, True)
            self.pipe = self._inferencer.pipe
            self.model = self._inferencer.model
            self.device = self._inferencer.device
            self.e_mode = self._inferencer.e_mode

    def convert_inputs(self, inputs):
        if self.input_style == 'audio_path, text':  # visual chatgpt style  # noqa
            audio_path = inputs.split(',')[0]
            prompt = ','.join(inputs.split(',')[1:])
            audio_path = audio_path.strip()
            prompt = prompt.strip()
            return audio_path, prompt
        else:
            raise NotImplementedError

    def apply(self, inputs):
        audio_path, prompt = inputs
        if self.remote:
            raise NotImplementedError
        else:
            if self.e_mode:
                self.pipe.to(self.device)
                self.model.to(self.device)

            audio_paths = [audio_path]
            text = ib.load_and_transform_text([prompt], self.device)
            embeddings = self.model.forward({ib.ModalityType.TEXT: text},
                                            normalize=False)
            text_embeddings = embeddings[ib.ModalityType.TEXT]

            audio_data = ib.load_and_transform_audio_data(
                audio_paths, self.device)
            embeddings = self.model.forward({
                ib.ModalityType.AUDIO: audio_data,
            })
            audio_embeddings = embeddings[ib.ModalityType.AUDIO]
            embeddings = text_embeddings * 0.5 + audio_embeddings * 0.5
            images = self.pipe(
                image_embeds=embeddings.half(), width=512, height=512).images
            new_img_name = get_new_image_name(audio_paths[0],
                                              'AudioText2Image')
            images[0].save(new_img_name)

            if self.e_mode:
                self.pipe.to('cpu')
                self.model.to('cpu')

        return new_img_name

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
