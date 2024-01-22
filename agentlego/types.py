from io import BytesIO, IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image
from typing_extensions import Annotated

from .schema import Parameter
from .utils import is_package_available, temp_path

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
            raise NotImplementedError(f'The value type `{type(value)}` is not '
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


class File(IOType):
    support_types = {'path': str, 'bytes': bytes}

    def __init__(self, value: Union[str, bytes], filetype: Optional[str] = None):
        super().__init__(value)
        if self.type == 'path' and not Path(self.value).exists():
            raise FileNotFoundError(f"No such file: '{self.value}'")
        self.filetype = filetype

    def to_path(self) -> str:
        return self.to('path')

    def to_bytes(self) -> bytes:
        return self.to('bytes')

    def to_file(self) -> IOBase:
        if self.type == 'path':
            return open(self.value, 'rb')
        else:
            return BytesIO(self.value)

    @classmethod
    def from_file(cls, file: IOBase, filetype: Optional[str] = None) -> 'File':
        return cls(file.read(), filetype=filetype)

    @staticmethod
    def _path_to_bytes(path: str) -> bytes:
        return open(path, 'rb').read()

    def _bytes_to_path(self, data: bytes) -> str:
        if self.filetype:
            category, _, suffix = self.filetype.partition('/')
            suffix = '.' + suffix if suffix else ''
        else:
            category = 'file'
            suffix = ''
        path = temp_path(category, suffix)
        with open(path, 'wb') as f:
            f.write(data)
        return path


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

    def to_file(self) -> IOBase:
        if self.type == 'path':
            return open(self.value, 'rb')
        else:
            file = BytesIO()
            self.to_pil().save(file, 'PNG')
            file.seek(0)
            return file

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

    def to_file(self) -> IOBase:
        if self.type == 'path' or not is_package_available('torchaudio'):
            return open(self.to_path(), 'rb')
        else:
            import torchaudio
            file = BytesIO()
            torchaudio.save(file, self.to_tensor(), self.sampling_rate)
            file.seek(0)
            return file

    @classmethod
    def from_file(cls, file: IOBase) -> 'AudioIO':
        try:
            import torchaudio
            audio, sr = torchaudio.load(file)
            return cls(audio, sampling_rate=sr)
        except ImportError:
            filename = temp_path('audio', '.wav')
            with open(filename, 'wb') as f:
                f.write(file.read())
            return cls(filename)

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


def Info(description: Optional[str] = None,
         *,
         name: Optional[str] = None,
         filetype: Optional[str] = None):
    """Used to add additional information of arguments and outputs.

    Args:
        description (str | None): Description for the parameter. Defaults to None.
        name (str | None): tool name for agent to identify the tool. Defaults to None.
        filetype (str | None): The file type for `File` inputs and outputs.
            Defaults to None.

    Examples:

        .. code:: python
        from agentlego.types import Annotated, Info, File

        class CustomTool(BaseTool):
            ...
            def apply(
                self, arg1: Annotated[str, Info('Description of arg1')]
            ) -> Annotated[File, Info('Description of output.', filetype='office/xlsx')]:
                pass
    """
    return Parameter(description=description, name=name, filetype=filetype)


CatgoryToIO = {
    'image': ImageIO,
    'text': str,
    'audio': AudioIO,
    'bool': bool,
    'int': int,
    'float': float,
    'file': File,
}

__all__ = ['ImageIO', 'AudioIO', 'CatgoryToIO', 'Info', 'Annotated']
