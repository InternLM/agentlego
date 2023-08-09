# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from typing import Union

import numpy as np
import torch
from mmengine import get
from mmengine.utils import apply_to

from ..base_tool import BaseTool
from ..parsers.type_mapping_parser import Audio


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
    DEFAULT_TOOLMETA = dict(
        name='Text Reader',
        model='microsoft/speecht5_tts',
        description='This is a tool that reads an English text out loud. It '
        'takes an {{{input:text}}} which should contain the'
        'text to read (in English) and returns a {{{output:audio}}} '
        'containing the sound.')
    default_speaker_embedding = 'https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_awb_arctic-wav-arctic_a0002.npy'  # noqa: E501

    def __init__(self,
                 *args,
                 post_processor: str = 'microsoft/speecht5_hifigan',
                 speaker_embeddings: Union[str, torch.Tensor, None] = None,
                 **kwargs):  # noqa: E501
        super().__init__(*args, **kwargs)
        self.post_processor = post_processor
        if speaker_embeddings is None:
            speaker_embeddings = self.default_speaker_embedding

        if isinstance(speaker_embeddings, str):
            with BytesIO(get(speaker_embeddings)) as f:
                speaker_embeddings = torch.from_numpy(np.load(f)).unsqueeze(0)
        self.speaker_embeddings = speaker_embeddings

    def setup(self) -> None:
        try:
            from transformers.models.speecht5 import (SpeechT5ForTextToSpeech,
                                                      SpeechT5HifiGan,
                                                      SpeechT5Processor)
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `transformers` correctly')
        self.pre_processor = SpeechT5Processor.from_pretrained(
            self.toolmeta.model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.toolmeta.model)
        self.post_processor = SpeechT5HifiGan.from_pretrained(
            self.post_processor)
        self.model.to(self.device)

    def apply(self, text: str) -> Audio:
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
        outputs = self.post_processor(outputs).cpu().detach()
        return Audio(array=outputs.numpy(), sampling_rate=self.SAMPLING_RATE)
