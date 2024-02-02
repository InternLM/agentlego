from urllib.parse import quote_plus

import requests

from agentlego.types import Annotated, Info
from ..base import BaseTool

LANG_CODES = {
    'zh-CN': 'Chinese',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'la': 'Latin',
    'pl': 'Polish',
    'ru': 'Russian',
    'es': 'Spanish',
    'th': 'Thai',
    'tr': 'Turkish',
}


class Translation(BaseTool):
    default_desc = ('This tool can translate a text from source language to '
                    'the target language. The language code should be one of ' +
                    ', '.join(f"'{k}' ({v})" for k, v in LANG_CODES.items()) + '.')

    def __init__(self, backend: str = 'google', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        if backend == 'google':
            self._translate = self.google_translate
        else:
            raise NotImplementedError(f'The backend {backend} is not available.')

    def apply(
        self,
        text: Annotated[str, Info('The text to translate')],
        target: Annotated[str, Info('The target language code')],
        source: Annotated[str, Info('The source language code')] = 'auto',
    ) -> str:
        return self._translate(text, target, source)

    def google_translate(self, text: str, target: str, source: str = 'auto') -> str:
        text = quote_plus(text)
        url_tmpl = ('https://translate.googleapis.com/translate_a/'
                    'single?client=gtx&sl={}&tl={}&dt=at&dt=bd&dt=ex&'
                    'dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&q={}')
        response = requests.get(url_tmpl.format(source, target, text), timeout=10).json()
        try:
            result = ''.join(x[0] for x in response[0] if x[0] is not None)
        except Exception:
            raise ConnectionError('Failed to translate.')

        return result
