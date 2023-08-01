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
            tool = load_tool('SpeechToTextTool', parser=parser, device='cuda')
            audio_path = osp.join(
                osp.dirname(__file__), '..', 'data', 'audio',
                'speech_to_text.flac')
            text = tool(audio_path)
            self.assertIn('going along slushy country', text)
            text = tool(audio=audio_path)
            self.assertIn('going along slushy country', text)
