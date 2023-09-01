import os.path as osp
from unittest import TestCase, skipIf

from mmengine import is_installed

from mmlmtools import load_tool
from mmlmtools.tools.parsers import (Audio, HuggingFaceAgentParser,
                                     LangChainParser)


@skipIf(not is_installed('transformers') or not is_installed('torchaudio'),
        'only test TestSpeechToTextTool when `transformers` is installed')
class TestTextToSpeech(TestCase):

    def test_call_langchain_agent(self):
        parser = LangChainParser()
        tool = load_tool('TextToSpeech', parser=parser, device='cuda')
        audio_path = tool('sing a song')
        osp.exists(audio_path)
        self.assertTrue(audio_path.endswith('.wav'))
        # Pass kwargs to the parser
        audio_path = tool(text='sing a song')
        self.assertTrue(audio_path.endswith('.wav'))
        osp.exists(audio_path)

    def test_call_huggingface_agent(self):
        parser = HuggingFaceAgentParser()
        tool = load_tool('TextToSpeech', parser=parser, device='cuda')
        audio = tool('sing a song')
        self.assertIsInstance(audio, Audio)

        # Pass kwargs to the parser
        audio = tool(text='sing a song')
        self.assertIsInstance(audio, Audio)
