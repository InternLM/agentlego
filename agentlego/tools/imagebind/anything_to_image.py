from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import AudioIO, ImageIO
from agentlego.utils import is_package_available, load_or_build_object, require
from ..base import BaseTool

if is_package_available('torch'):
    import torch


class AnythingToImage:

    @require(['diffusers', 'ftfy', 'iopath', 'timm'])
    def __init__(self, device):
        from diffusers import StableUnCLIPImg2ImgPipeline

        from .models.imagebind_model import imagebind_huge

        pipe = load_or_build_object(
            StableUnCLIPImg2ImgPipeline.from_pretrained,
            pretrained_model_name_or_path='stabilityai/'
            'stable-diffusion-2-1-unclip',
            torch_dtype=torch.float16,
            variant='fp16')

        self.device = device
        self.pipe = pipe.to(device)
        self.pipe.enable_vae_slicing()
        self.model = imagebind_huge(pretrained=True).to(self.device)
        self.model.eval()


class AudioToImage(BaseTool):
    """A tool to generate image from an audio.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='AudioToImage',
        description=('This tool can generate an image '
                     'according to the input audio'),
        inputs=['audio'],
        outputs=['image'],
    )

    @require(['diffusers', 'ftfy', 'iopath', 'timm', 'pytorchvideo'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device)

    def apply(self, audio: AudioIO) -> ImageIO:
        from .data import load_and_transform_audio_data
        from .models.imagebind_model import ModalityType

        audio_paths = [audio.to_path()]
        audio_data = load_and_transform_audio_data(audio_paths, self.device)
        embeddings = self._inferencer.model.forward(
            {ModalityType.AUDIO: audio_data})
        embeddings = embeddings[ModalityType.AUDIO]
        images = self._inferencer.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        output_image = images[0]

        return ImageIO(output_image)


class ThermalToImage(BaseTool):
    """A tool to generate image from an thermal image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='ThermalToImage',
        description=('This tool can generate an image '
                     'according to the input thermal image.'),
        inputs=['image'],
        outputs=['image'],
    )

    @require(['diffusers', 'ftfy', 'iopath', 'timm'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device)

    def apply(self, thermal: ImageIO) -> ImageIO:
        from .data import load_and_transform_thermal_data
        from .models.imagebind_model import ModalityType

        thermal_paths = [thermal.to_path()]
        thermal_data = load_and_transform_thermal_data(thermal_paths,
                                                       self.device)
        embeddings = self._inferencer.model.forward(
            {ModalityType.THERMAL: thermal_data})
        embeddings = embeddings[ModalityType.THERMAL]
        images = self._inferencer.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        output_image = images[0]

        return ImageIO(output_image)


class AudioImageToImage(BaseTool):
    """A tool to generate image from an audio and an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='AudioImageToImage',
        description=('This tool can generate an image according to '
                     'the input reference image and the input audio.'),
        inputs=['image', 'audio'],
        outputs=['image'],
    )

    @require(['diffusers', 'ftfy', 'iopath', 'timm', 'pytorchvideo'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device)

    def apply(self, image: ImageIO, audio: AudioIO) -> ImageIO:
        from .data import (load_and_transform_audio_data,
                           load_and_transform_vision_data)
        from .models.imagebind_model import ModalityType

        # process image data
        vision_data = load_and_transform_vision_data([image.to_path()],
                                                     self.device)
        embeddings = self._inferencer.model.forward(
            {ModalityType.VISION: vision_data}, normalize=False)
        img_embeddings = embeddings[ModalityType.VISION]

        # process audio data
        audio_data = load_and_transform_audio_data([audio.to_path()],
                                                   self.device)
        embeddings = self._inferencer.model.forward({
            ModalityType.AUDIO:
            audio_data,
        })
        audio_embeddings = embeddings[ModalityType.AUDIO]

        embeddings = (img_embeddings + audio_embeddings) / 2
        images = self._inferencer.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        output_image = images[0]

        return ImageIO(output_image)


class AudioTextToImage(BaseTool):
    """A tool to generate image from an audio and texts.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='AudioTextToImage',
        description=('This tool can generate an image according to '
                     'the input audio and the input description.'),
        inputs=['audio', 'text'],
        outputs=['image'],
    )

    @require(['diffusers', 'ftfy', 'iopath', 'timm', 'pytorchvideo'])
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.device = device

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device)

    def apply(self, audio: AudioIO, prompt: str) -> ImageIO:
        from .data import (load_and_transform_audio_data,
                           load_and_transform_text)
        from .models.imagebind_model import ModalityType

        audio_paths = [audio.to_path()]
        text = load_and_transform_text([prompt], self.device)
        embeddings = self._inferencer.model.forward({ModalityType.TEXT: text},
                                                    normalize=False)
        text_embeddings = embeddings[ModalityType.TEXT]

        audio_data = load_and_transform_audio_data(audio_paths, self.device)
        embeddings = self._inferencer.model.forward({
            ModalityType.AUDIO:
            audio_data,
        })
        audio_embeddings = embeddings[ModalityType.AUDIO]
        embeddings = text_embeddings * 0.5 + audio_embeddings * 0.5
        images = self._inferencer.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        output_image = images[0]

        return ImageIO(output_image)
