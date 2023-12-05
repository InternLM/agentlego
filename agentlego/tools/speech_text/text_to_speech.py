# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from typing import Callable, Union

import requests

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import AudioIO
from agentlego.utils import is_package_available, require
from ..base import BaseTool

if is_package_available('torch'):
    import torch


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
        speaker_embeddings (str | dict): The speaker embedding
            of the Speech-T5 model. Defaults to an embedding from
            ``Matthijs/speecht5-tts-demo``.
        device (str): The device to load the model. Defaults to 'cuda'.
    """
    SAMPLING_RATE = 16000
    DEFAULT_TOOLMETA = ToolMeta(
        name='Text Reader',
        description='This is a tool that can speak the input text into audio.',
        inputs=['text'],
        outputs=['audio'],
    )

    @require('TTS', 'langid')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
                 speaker_embeddings: Union[str, dict] = (
                     'http://download.openmmlab.com/agentlego/'
                     'default_voice.pth'),
                 sampling_rate=16000,
                 device='cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model_name = model

        if isinstance(speaker_embeddings, str):
            with BytesIO(requests.get(speaker_embeddings).content) as f:
                speaker_embeddings = torch.load(f, map_location=device)
        self.speaker_embeddings = speaker_embeddings
        self.sampling_rate = sampling_rate
        self.device = device

    def setup(self) -> None:
        from TTS.api import TTS
        from TTS.tts.models.xtts import Xtts
        self.model = TTS(self.model_name).to(self.device).synthesizer.tts_model
        self.model: Xtts

    def apply(self, text: str) -> AudioIO:
        import langid
        langid.set_languages([
            lang if lang != 'zh-cn' else 'zh'
            for lang in self.model.config.languages
        ])
        lang = langid.classify(text)[0]
        lang = 'zh-cn' if lang == 'zh' else lang
        text = text.replace('，', ', ').replace('。', '. ').replace(
            '？', '? ').replace('！', '! ').replace('、', ', ').strip()
        out = self.model.inference(
            text,
            language=lang,
            do_sample=False,
            enable_text_splitting=len(text) > 72,  # Split text if too long.
            **self.speaker_embeddings,
        )

        return AudioIO(
            torch.tensor(out['wav']).unsqueeze(0), sampling_rate=24000)
