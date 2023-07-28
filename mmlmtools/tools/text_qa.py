# Copyright (c) OpenMMLab. All rights reserved.
import re

from ..utils.toolmeta import ToolMeta
from .base_tool_v1 import BaseToolv1


class TextQuestionAnsweringTool(BaseToolv1):
    DEFAULT_TOOLMETA = dict(
        name='Text Question Answering',
        model={'model_name': 'google/flan-t5-base'},
        description='This is a useful tool that answers questions related to '
        'given texts.',
        input_description='It takes a strings as the input, which contains '
        'the text and the question. The question should be ended with a '
        'question mark',
        output_description='It returns a string as the output, representing '
        'the answer to the question.')

    QA_PROMPT = ('Here is a text containing a lot of information: {text} '
                 'Can you answer this question about the text: {question}')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'text',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._tokenizer = None
        self._model = None

    def setup(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError:
                raise ImportError('TextQuestionAnswering tool requires '
                                  'transformers package, please install it '
                                  'with `pip install transformers`.')

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.toolmeta.model['model_name'])
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.toolmeta.model['model_name'])

            self._model.to(self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'text':
            if isinstance(inputs, str):
                match = re.search(r'([^.;]*?\?)', inputs)
                if match is None:
                    text, question = inputs, inputs
                else:
                    text, question = inputs, match.group(0).strip()
            else:
                assert isinstance(inputs, tuple) and len(inputs) == 2
                text, question = inputs
            return text, question

        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            text, question = self.convert_inputs(inputs)
            prompt = self.QA_PROMPT.format(text=text, question=question)
            encoded_input = self._tokenizer(prompt, return_tensors='pt')

            output_ids = self._model.generate(**encoded_input)
            in_b, _ = encoded_input['input_ids'].shape
            out_b = output_ids.shape[0]
            output = output_ids.reshape(in_b, out_b // in_b,
                                        *output_ids.shape[1:])[0][0]

            answer = self._tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            return answer
