# Copyright (c) OpenMMLab. All rights reserved.
import requests
from typing import Callable, Union
from urllib.parse import quote_plus

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from ..base import BaseTool

LANG_CODES = {
    "auto": "Detect source language",
    "zh-CN": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "la": "Latin",
    "pl": "Polish",
    "ru": "Russian",
    "es": "Spanish",
    "th": "Thai",
    "tr": "Turkish",
}


class Translation(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Text translation',
        description='This tool can translate a text from source language to '
        'the target language. The source_lang and target_lang can be one of ' +
        ', '.join(f"'{k}' ({v})" for k, v in LANG_CODES.items()) + '.',
        inputs=['text', 'text', 'text'],
        outputs=['text'],
    )

    PROMPT = ('translate {source_lang} to {target_lang}: {input}')

    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 backend: str = 'google'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        if backend == 'google':
            self._translate = self.google_translate
        else:
            raise NotImplementedError(
                f'The backend {backend} is not available.')

    def apply(self, text: str, source_lang: str, target_lang: str) -> str:
        return self._translate(text, source_lang, target_lang)

    def google_translate(self, text: str, source: str, target: str):
        text = quote_plus(text)
        url_tmpl = ('https://translate.googleapis.com/translate_a/'
                    'single?client=gtx&sl={}&tl={}&dt=at&dt=bd&dt=ex&'
                    'dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&q={}')
        response = requests.get(
            url_tmpl.format(source, target, text), timeout=10).json()
        try:
            result = ''.join(x[0] for x in response[0] if x[0] is not None)
        except:
            raise ConnectionError('Failed to translate.')

        return result
