# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from mmlmtools.utils import get_new_file_path


@dataclass
class Audio:
    array: np.ndarray
    sampling_rate: str
    path: Optional[str] = None

    def _ipython_display_(self, include=None, exclude=None):
        import torchaudio
        from IPython.display import Audio, display
        if self.path is None:
            path = get_new_file_path(
                osp.join('data', 'audio', 'temp.wav'),
                func_name='_ndarray_to_audio_path')
            self.path = path
        torchaudio.save(path,
                        torch.from_numpy(self.array).reshape(1, -1).float(),
                        self.sampling_rate)

        display(Audio(path, rate=self.sampling_rate))

    @classmethod
    def from_path(cls, path):
        import torchaudio
        audio = torchaudio.load(path)
        return cls(
            path=path,
            array=audio[0].reshape(-1).numpy(),
            sampling_rate=audio[1],
        )
