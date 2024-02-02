from agentlego.types import AudioIO
from agentlego.utils import apply_to, is_package_available, load_or_build_object, require
from ..base import BaseTool

if is_package_available('torch'):
    import torch

if is_package_available('torchaudio'):
    import torchaudio


def resampling_audio(audio: AudioIO, new_rate):
    tensor, ori_sampling_rate = audio.to_tensor(), audio.sampling_rate
    tensor = torchaudio.functional.resample(tensor, ori_sampling_rate, new_rate)
    return AudioIO(tensor, sampling_rate=new_rate)


class SpeechToText(BaseTool):
    """A tool to recognize speech and convert to text.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``HuggingFace`` model page.
            Defaults to ``openai/whisper-base``.
        device (str): The device to load the model. Defaults to 'cpu'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'The tool can translate spoken language audio into text.'

    @require(('torch', 'transformers', 'torchaudio'))
    def __init__(self, model='openai/whisper-base', device='cuda', toolmeta=None):
        super().__init__(toolmeta)
        self.model_name = model
        self.device = device

    def setup(self) -> None:
        from transformers.models.whisper import (WhisperForConditionalGeneration,
                                                 WhisperProcessor)
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
        encoded_inputs = apply_to(encoded_inputs, lambda x: isinstance(x, torch.Tensor),
                                  lambda x: x.to(self.device))
        outputs = self.model.generate(inputs=encoded_inputs)
        outputs = apply_to(outputs, lambda x: isinstance(x, torch.Tensor),
                           lambda x: x.to('cpu'))
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
