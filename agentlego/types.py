# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image

from .utils import temp_path

if TYPE_CHECKING:
    import torch


def _typename(obj):
    return f'{type(obj).__module__}.{type(obj).__name__}'


class IOType:
    support_types = {}

    def __init__(self, value):
        if type(value).__qualname__ == 'AgentType':
            # Handle hugginface agent
            value = value._value

        self.type = None
        for name, cls in self.support_types.items():
            if isinstance(cls, type) and isinstance(value, cls):
                self.value = value
                self.type = name
            elif isinstance(cls, str) and _typename(value) == cls:
                self.value = value
                self.type = name
        if self.type is None:
            raise NotImplementedError(
                f'The value type `{type(value)}` is not '
                f'supported by `{self.__class__.__name__}`')

    def to(self, dst_type: str):
        if self.type == dst_type:
            return self.value

        assert dst_type in self.support_types
        convert_fn = f'_{self.type}_to_{dst_type}'
        assert hasattr(self, convert_fn)

        return getattr(self, convert_fn)(self.value)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'


class ImageIO(IOType):
    support_types = {'path': str, 'pil': Image.Image, 'array': np.ndarray}

    def __init__(self, value: Union[str, np.ndarray, Image.Image]):
        super().__init__(value)
        if self.type == 'path' and not Path(self.value).exists():
            raise FileNotFoundError(f"No such file: '{self.value}'")

    def to_path(self) -> str:
        return self.to('path')

    def to_pil(self) -> Image.Image:
        return self.to('pil')

    def to_array(self) -> np.ndarray:
        return self.to('array')

    @staticmethod
    def _path_to_pil(path: str) -> Image.Image:
        return Image.open(path)

    @staticmethod
    def _path_to_array(path: str) -> np.ndarray:
        return np.array(Image.open(path).convert('RGB'))

    @staticmethod
    def _pil_to_path(image: Image.Image) -> str:
        filename = temp_path('image', '.png')
        image.save(filename)
        return filename

    @staticmethod
    def _pil_to_array(image: Image.Image) -> np.ndarray:
        return np.array(image.convert('RGB'))

    @staticmethod
    def _array_to_pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    @staticmethod
    def _array_to_path(image: np.ndarray) -> str:
        filename = temp_path('image', '.png')
        Image.fromarray(image).save(filename)
        return filename


class AudioIO(IOType):
    DEFAULT_SAMPLING_RATE = 16000
    support_types = {'tensor': 'torch.Tensor', 'path': str}

    def __init__(self,
                 value: Union[np.ndarray, 'torch.Tensor', str],
                 sampling_rate: Optional[int] = None):
        if type(value).__qualname__ == 'AgentAudio':
            # Handle hugginface agent
            sampling_rate = value.samplerate
            value = value.to_raw()

        super().__init__(value=value)
        self._sampling_rate = sampling_rate

        if self.type == 'path' and not Path(self.value).exists():
            raise FileNotFoundError(f"No such file: '{self.value}'")

    @property
    def sampling_rate(self) -> int:
        if self._sampling_rate is not None:
            return self._sampling_rate
        elif self.type == 'path':
            self.to('tensor')
            return self._sampling_rate
        else:
            return self.DEFAULT_SAMPLING_RATE

    def to_tensor(self) -> 'torch.Tensor':
        return self.to('tensor')

    def to_path(self) -> str:
        return self.to('path')

    def _path_to_tensor(self, path: str) -> 'torch.Tensor':
        import torchaudio
        audio, sampling_rate = torchaudio.load(path)
        self._sampling_rate = sampling_rate
        return audio

    def _tensor_to_path(self, tensor: 'torch.Tensor') -> str:
        import torchaudio
        filename = temp_path('audio', '.wav')
        torchaudio.save(filename, tensor, self.sampling_rate)
        return filename


CatgoryToIO = {'image': ImageIO, 'text': str, 'audio': AudioIO}

__all__ = ['ImageIO', 'AudioIO', 'CatgoryToIO']
