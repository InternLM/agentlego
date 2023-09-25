from unittest import skipIf

from mmengine import is_installed

from agentlego import load_tool
from agentlego.testing import ToolTestCase


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudioToImage(ToolTestCase):

    def test_call(self):
        tool = load_tool('AudioToImage', device='cpu')
        res = tool('tests/data/audio/cat.wav')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestThermalToImage(ToolTestCase):

    def test_call(self):
        tool = load_tool('ThermalToImage', device='cpu')
        res = tool('tests/data/thermal/030419.jpg')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudioImageToImage(ToolTestCase):

    def test_call(self):
        tool = load_tool('AudioImageToImage', device='cpu')
        res = tool('tests/data/images/dog.jpg, tests/data/audio/cat.wav')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestAudioTextToImage(ToolTestCase):

    def test_call(self):
        tool = load_tool('AudioTextToImage', device='cpu')
        res = tool(
            'tests/data/audio/cat.wav, generate a cat flying in the sky')
        assert isinstance(res, str)
