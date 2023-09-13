# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from typing import Callable, Union

import numpy as np
import torch
from mmengine import get
from mmengine.utils import apply_to

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import AudioIO
from mmlmtools.utils import require
from ..base import BaseTool


def resampling_audio(audio: dict, new_rate):
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(f'Failed to run the tool: {e} '
                          '`torchaudio` is not installed correctly')
    array, ori_sampling_rate = audio['array'], audio['sampling_rate']
    array = torch.from_numpy(array).reshape(-1, 1)
    torchaudio.functional.resample(array, ori_sampling_rate, new_rate)
    return {
        'array': array.reshape(-1).numpy(),
        'sampling_rate': new_rate,
        'path': audio['path']
    }


class TextToSpeech(BaseTool):
    SAMPLING_RATE = 16000
    DEFAULT_TOOLMETA = ToolMeta(
        name='Text Reader',
        description='This is a tool that can speak the input text into audio.',
        inputs=['text'],
        outputs=['audio'],
    )

    @require('transformers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 post_processor: str = 'microsoft/speecht5_hifigan',
                 model='microsoft/speecht5_tts',
                 speaker_embeddings: Union[str, torch.Tensor] = (
                     'https://huggingface.co/spaces/Matthijs/'
                     'speecht5-tts-demo/resolve/main/spkemb/'
                     'cmu_us_awb_arctic-wav-arctic_a0002.npy'),
                 sampling_rate=16000,
                 device='cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.post_processor_name = post_processor
        self.model_name = model

        if isinstance(speaker_embeddings, str):
            with BytesIO(get(speaker_embeddings)) as f:
                speaker_embeddings = torch.from_numpy(np.load(f)).unsqueeze(0)
        self.speaker_embeddings = speaker_embeddings
        self.sampling_rate = sampling_rate
        self.device = device

    def setup(self) -> None:
        from transformers.models.speecht5 import (SpeechT5ForTextToSpeech,
                                                  SpeechT5HifiGan,
                                                  SpeechT5Processor)
        self.pre_processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_name).to(self.device)
        self.post_processor = SpeechT5HifiGan.from_pretrained(
            self.post_processor_name)

    def apply(self, text: str) -> AudioIO:
        encoded_inputs = self.pre_processor(
            text=text, return_tensors='pt', truncation=True)
        encoded_inputs = dict(
            input_ids=encoded_inputs['input_ids'],
            speaker_embeddings=self.speaker_embeddings)
        encoded_inputs = apply_to(encoded_inputs,
                                  lambda x: isinstance(x, torch.Tensor),
                                  lambda x: x.to(self.device))
        outputs = self.model.generate_speech(**encoded_inputs)
        outputs = apply_to(outputs, lambda x: isinstance(x, torch.Tensor),
                           lambda x: x.to('cpu'))
        outputs = self.post_processor(outputs).cpu().detach().reshape(1, -1)
        return AudioIO(outputs, sampling_rate=self.sampling_rate)
