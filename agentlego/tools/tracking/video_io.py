from agentlego.types import IOType, ImageIO
from agentlego.utils.file import temp_path
from io import BytesIO, IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image
from typing_extensions import Annotated

import os 

class VideoIO(IOType):
    support_types = {'path': str}

    def __init__(self, value: str):
        super().__init__(value)
        if self.type == 'path' and not Path(self.value).exists():
            raise FileNotFoundError(f"No such file: '{self.value}'")
        
        self.root_path = Path(self.value)
        self.images = sorted(os.listdir(self.root_path))
        self.cnt = 0

    def to_path(self) -> str:
        return self.to('path')

    def to_pil(self) -> Image.Image:
        return self.to('pil')

    def to_array(self) -> np.ndarray:
        return self.to('array')

    def to_file(self) -> IOBase:
        if self.type == 'path':
            return open(self.value, 'rb')
        else:
            file = BytesIO()
            self.to_pil().save(file, 'PNG')
            file.seek(0)
            return file
        
    def next_image(self) -> Image:
        ret = Image.open(self.images[self.cnt])
        self.cnt += 1
        return ret 
    
    def is_finish(self) -> bool:
        return self.cnt == len(self.images) - 1

    @classmethod
    def from_file(cls, file: IOBase) -> 'ImageIO':
        from PIL import Image
        return cls(Image.open(file))

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
    

    