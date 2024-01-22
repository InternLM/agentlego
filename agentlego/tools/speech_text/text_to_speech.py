from io import BytesIO
from typing import Union

import requests

from agentlego.types import Annotated, AudioIO, Info
from agentlego.utils import is_package_available, require
from ..base import BaseTool

if is_package_available('torch'):
    import torch

LANG_CODES = {
    'zh-cn': 'Chinese',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'tr': 'Turkish',
    'ru': 'Russian',
    'ar': 'Arabic',
    'ja': 'Japanese',
    'ko': 'Korean',
    # "pt": "Portuguese",
    # "pl": "Polish",
    # "nl": "Dutch",
    # "cs": "Czech",
    # "hu": "Hungarian",
    # "hi": "Hindi",
}


class TextToSpeech(BaseTool):
    """A tool to convert input text to speech audio.

    Args:
        model (str): The model name used to inference. Which can be found
            in https://github.com/coqui-ai/TTSHuggingFace .
            Defaults to ``tts_models/multilingual/multi-dataset/xtts_v2``.
        speaker_embeddings (str | dict): The speaker embedding
            of the TTS model. Defaults to a default embedding.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    SPEAKER_EMBEDDING = ('http://download.openmmlab.com/agentlego/default_voice.pth')
    default_desc = ('The tool can speak the input text into audio. The language code '
                    'should be one of ' +
                    ', '.join(f"'{k}' ({v})" for k, v in LANG_CODES.items()) + '.')

    @require('TTS', 'langid')
    def __init__(self,
                 model: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
                 speaker_embeddings: Union[str, dict] = SPEAKER_EMBEDDING,
                 device='cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model_name = model

        if isinstance(speaker_embeddings, str):
            with BytesIO(requests.get(speaker_embeddings).content) as f:
                speaker_embeddings = torch.load(f, map_location=device)
        self.speaker_embeddings = speaker_embeddings
        self.device = device

    def setup(self) -> None:
        from TTS.api import TTS
        from TTS.tts.models.xtts import Xtts
        self.model = TTS(self.model_name).to(self.device).synthesizer.tts_model
        self.model: Xtts

    def apply(
        self,
        text: str,
        lang: Annotated[str, Info('The language code of text.')] = 'auto',
    ) -> AudioIO:
        if lang == 'auto':
            import langid
            langid.set_languages(
                [lang if lang != 'zh-cn' else 'zh' for lang in LANG_CODES])
            lang = langid.classify(text)[0]
            lang = 'zh-cn' if lang == 'zh' else lang

        text = text.replace('，', ', ').replace('。', '. ').replace('？', '? ').replace(
            '！', '! ').replace('、', ', ').strip()
        out = self.model.inference(
            text,
            language=lang,
            do_sample=False,
            enable_text_splitting=len(text) > 72,  # Split text if too long.
            **self.speaker_embeddings,
        )

        return AudioIO(torch.tensor(out['wav']).unsqueeze(0), sampling_rate=24000)
