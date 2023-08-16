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
            tool = load_tool('TextToSpeech', parser=parser, device='cuda')
            audio = tool('sing a song')
            audio_path = audio if isinstance(audio, str) else audio.path
            osp.exists(audio_path)
            self.assertTrue(audio_path.endswith('.wav'))
            # Pass kwargs to the parser
            audio = tool(text='sing a song')
            audio_path = audio if isinstance(audio, str) else audio.path
            self.assertTrue(audio_path.endswith('.wav'))
            osp.exists(audio_path)
