# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.utils import apply_to

from ..base_tool import BaseTool
from ..parsers.type_mapping_parser import Audio


def resampling_audio(audio: Audio, new_rate):
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(f'Failed to run the tool: {e} '
                          '`torchaudio` is not installed correctly')
    array, ori_sampling_rate = audio.array, audio.sampling_rate
    array = torch.from_numpy(array).reshape(-1, 1)
    torchaudio.functional.resample(array, ori_sampling_rate, new_rate)
    return Audio(
        array=array.reshape(-1).numpy(),
        sampling_rate=new_rate,
        path=audio.path)


class SpeechToText(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Transcriber',
        model='openai/whisper-base',
        description='This is a tool that transcribes an audio into text. It '
        'It takes an {{{input:audio}}} as the input, and returns a '
        '{{{output:text}}} representing the description of the audio')

    def setup(self) -> None:
        try:
            from transformers.models.whisper import (
                WhisperForConditionalGeneration, WhisperProcessor)
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `transformers` correctly')
        self.processor = WhisperProcessor.from_pretrained(self.toolmeta.model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.toolmeta.model)
        self.model.to(self.device)

    def apply(self, audio: Audio) -> str:
        target_sampling_rate = self.processor.feature_extractor.sampling_rate
        if target_sampling_rate != audio.sampling_rate:
            audio = resampling_audio(audio, target_sampling_rate)
        encoded_inputs = self.processor(
            audio.array, return_tensors='pt').input_features
        encoded_inputs = apply_to(encoded_inputs,
                                  lambda x: isinstance(x, torch.Tensor),
                                  lambda x: x.to(self.device))
        outputs = self.model.generate(inputs=encoded_inputs)
        outputs = apply_to(outputs, lambda x: isinstance(x, torch.Tensor),
                           lambda x: x.to('cpu'))
        return self.processor.batch_decode(
            outputs, skip_special_tokens=True)[0]
