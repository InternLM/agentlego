import os.path as osp
from unittest import TestCase, skipIf

from mmengine import is_installed

from mmlmtools import load_tool
from mmlmtools.tools.parsers import HuggingFaceAgentParser, LangchainParser


@skipIf(not is_installed('transformers') or not is_installed('torchaudio'),
        'only test TestSpeechToTextTool when `transformers` is installed')
class TestSpeechToTextTool(TestCase):

    def test_call_langchain_agent(self):
        for parser in (LangchainParser(), HuggingFaceAgentParser()):
            tool = load_tool('TextToSpeechTool', parser=parser, device='cuda')
            audio_path = tool('sing a song')
            osp.exists(audio_path)
            self.assertTrue(audio_path.endswith('.wav'))

            audio_path = tool(text='sing a song')
            self.assertTrue(audio_path.endswith('.wav'))
            osp.exists(audio_path)
