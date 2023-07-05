from unittest import skipIf

from mmengine import is_installed

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudio2ImageTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('Audio2ImageTool', device='cuda')
        res = tool('tests/data/audio/cat.wav')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestThermal2ImageTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('Thermal2ImageTool', device='cuda')
        res = tool('tests/data/thermal/030419.jpg')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudioImage2ImageTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('AudioImage2ImageTool', device='cuda')
        res = tool('tests/data/images/dog_image.jpg, tests/data/audio/cat.wav')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudioText2ImageTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('AudioText2ImageTool', device='cuda')
        res = tool(
            'tests/data/audio/cat.wav, generate a cat flying in the sky')
        assert isinstance(res, str)
