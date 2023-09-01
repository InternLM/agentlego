# Copyright (c) OpenMMLab. All rights reserved.
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import numpy as np
import torch
import torchaudio
from PIL import Image


def _temp_path(category: str, suffix: str, root: str = 'generated'):
    output_dir = Path(root) / category
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = datetime.now().strftime('%Y%m%d') + '-' + str(uuid.uuid4())[:4]
    path = (output_dir / filename).with_suffix(suffix)
    return str(path.absolute())


class IOType:
    support_types = {}

    def __init__(self, value):
        if value.__class__.__qualname__ == 'AgentType':
            # Handle hugginface agent
            value = value._value

        self.type = None
        for name, cls in self.support_types.items():
            if isinstance(value, cls):
                self.value = value
                self.type = name
        if self.type is None:
            raise NotImplementedError()

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
        image = cv.imread(path)
        if image is None:
            raise RuntimeError(f'Failed to read {path}')
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)

    @staticmethod
    def _pil_to_path(image: Image.Image) -> str:
        filename = _temp_path('image', '.png')
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
        filename = _temp_path('image', '.png')
        cv.imwrite(filename, cv.cvtColor(image, cv.COLOR_RGB2BGR))
        return filename


class AudioIO(IOType):
    DEFAULT_SAMPLING_RATE = 16000
    support_types = {'tensor': torch.Tensor, 'path': str}

    def __init__(self,
                 value: Union[torch.Tensor, str],
                 sampling_rate: Optional[int] = None):
        if value.__class__.__qualname__ == 'AgentAudio':
            # Handle hugginface agent
            sampling_rate = value.sampling_rate
            value = value.to_raw()

        super().__init__(value=value)
        self._sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> int:
        if self._sampling_rate is not None:
            return self._sampling_rate
        elif self.type == 'path':
            self.to('tensor')
            return self._sampling_rate
        else:
            return self.DEFAULT_SAMPLING_RATE

    def to_tensor(self) -> torch.Tensor:
        return self.to('tensor')

    def to_path(self) -> str:
        return self.to('path')

    def _path_to_tensor(self, path: str) -> torch.Tensor:
        audio, sampling_rate = torchaudio.load(path)
        self._sampling_rate = sampling_rate
        return audio

    def _tensor_to_path(self, tensor: torch.Tensor) -> str:
        filename = _temp_path('audio', '.wav')
        torchaudio.save(filename, tensor, self.sampling_rate)
        return filename


CatgoryToIO = {'image': ImageIO, 'text': str, 'audio': AudioIO}

__all__ = ['ImageIO', 'AudioIO', 'CatgoryToIO']
