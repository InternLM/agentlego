# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmlmtools.utils.toolmeta import ToolMeta
from .base_tool import BaseTool
from .parsers import BaseParser


class TextTranslation(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Text translation',
        model={'model_name': 't5-small'},
        description='This is a useful tool that converts a text from one '
        'language to another. It takes three arguments as inputs: the '
        '{{{input:text}}} to be translated, the source language '
        '{{{source_lang:text}}} and the target language '
        '{{{target_lang:text}}}. It retures the translated text.',
    )

    PROMPT = ('translate {source_lang} to {target_lang}: {input}')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cpu'):
        super().__init__(toolmeta, parser, remote, device)

        self._tokenizer = None
        self._model = None

    def setup(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError:
                raise ImportError('TextTranslationBool tool requires '
                                  'transformers package, please install it '
                                  'with `pip install transformers`.')
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.toolmeta.model['model_name'])
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.toolmeta.model['model_name'])

            self._model.to(self.device)

    def apply(self, input: str, source_lang: str, target_lang: str):
        if self.remote:
            raise NotImplementedError
        else:
            text = self.PROMPT.format(
                source_lang=source_lang, target_lang=target_lang, input=input)
            encoded_input = self._tokenizer(
                text, return_tensors='pt').input_ids.to(self.device)
            outputs = self._model.generate(encoded_input)
            output = self._tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            return output
