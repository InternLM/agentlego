# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmengine.utils import apply_to

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import AudioIO
from mmlmtools.utils import require
from mmlmtools.utils.cache import load_or_build_object
from ..base import BaseTool


def resampling_audio(audio: AudioIO, new_rate):
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(f'Failed to run the tool: {e} '
                          '`torchaudio` is not installed correctly')
    tensor, ori_sampling_rate = audio.to_tensor(), audio.sampling_rate
    tensor = torchaudio.functional.resample(tensor, ori_sampling_rate,
                                            new_rate)
    return AudioIO(tensor, sampling_rate=new_rate)


class SpeechToText(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Transcriber',
        description='This is a tool that transcribes an audio into text.',
        inputs=['audio'],
        outputs=['text'],
    )

    @require('transformers')
    def __init__(
        self,
        toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
        parser: callable = DefaultParser,
        model='openai/whisper-base',
        device='cuda',
    ):
        super().__init__(toolmeta, parser)
        self.model_name = model
        self.device = device

    def setup(self) -> None:
        from transformers.models.whisper import (
            WhisperForConditionalGeneration, WhisperProcessor)
        self.processor = load_or_build_object(WhisperProcessor.from_pretrained,
                                              self.model_name)
        self.model = load_or_build_object(
            WhisperForConditionalGeneration.from_pretrained,
            self.model_name).to(self.device)

    def apply(self, audio: AudioIO) -> str:
        target_sampling_rate = self.processor.feature_extractor.sampling_rate
        if target_sampling_rate != audio.sampling_rate:
            audio = resampling_audio(audio, target_sampling_rate)
        encoded_inputs = self.processor(
            audio.to_tensor().numpy().reshape(-1),
            return_tensors='pt',
            sampling_rate=target_sampling_rate).input_features
        encoded_inputs = apply_to(encoded_inputs,
                                  lambda x: isinstance(x, torch.Tensor),
                                  lambda x: x.to(self.device))
        outputs = self.model.generate(inputs=encoded_inputs)
        outputs = apply_to(outputs, lambda x: isinstance(x, torch.Tensor),
                           lambda x: x.to('cpu'))
        return self.processor.batch_decode(
            outputs, skip_special_tokens=True)[0]
