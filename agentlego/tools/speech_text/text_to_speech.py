# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from typing import Callable, Union

import numpy as np
import torch
from mmengine import get
from mmengine.utils import apply_to

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import AudioIO
from agentlego.utils import require
from ..base import BaseTool


def resampling_audio(audio: dict, new_rate):
    import torchaudio
    array, ori_sampling_rate = audio['array'], audio['sampling_rate']
    array = torch.from_numpy(array).reshape(-1, 1)
    torchaudio.functional.resample(array, ori_sampling_rate, new_rate)
    return {
        'array': array.reshape(-1).numpy(),
        'sampling_rate': new_rate,
        'path': audio['path']
    }


class TextToSpeech(BaseTool):
    """A tool to convert input text to speech audio.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``HuggingFace`` model page.
            Defaults to ``microsoft/speecht5_tts``.
        post_processor (str): The post-processor of the output audio.
            Defaults to ``microsoft/speecht5_hifigan``.
        speaker_embeddings (str | torch.Tensor): The speaker embedding
            of the Speech-T5 model. Defaults to an embedding from
            ``Matthijs/speecht5-tts-demo``.
        device (str): The device to load the model. Defaults to 'cuda'.
    """

    SAMPLING_RATE = 16000
    DEFAULT_TOOLMETA = ToolMeta(
        name='Text Reader',
        description='This is a tool that can '
        'speak the input English text into audio.',
        inputs=['text'],
        outputs=['audio'],
    )

    @require('transformers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'microsoft/speecht5_tts',
                 post_processor: str = 'microsoft/speecht5_hifigan',
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
