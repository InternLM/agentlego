# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class Translation(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Text translation',
        description='This is a useful tool that translates a text from one '
        'language to another.',
        inputs=['text', 'text', 'text'],
        outputs=['text'],
    )

    PROMPT = ('translate {source_lang} to {target_lang}: {input}')

    @require('transformers')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 't5-small',
                 device: str = 'cuda'):
        super().__init__(toolmeta=toolmeta, parser=parser)

        self._tokenizer = None
        self._model_name = model
        self._model = None
        self._tokenizer = None
        self.device = device

    def setup(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = load_or_build_object(
            AutoTokenizer.from_pretrained,
            self._model_name,
        )
        self._model = load_or_build_object(
            AutoModelForSeq2SeqLM.from_pretrained,
            self._model_name,
        ).to(self.device)

    def apply(self, input: str, source_lang: str, target_lang: str) -> str:
        text = self.PROMPT.format(
            source_lang=source_lang, target_lang=target_lang, input=input)
        encoded_input = self._tokenizer(
            text, return_tensors='pt').input_ids.to(self.device)
        outputs = self._model.generate(encoded_input)
        output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output
