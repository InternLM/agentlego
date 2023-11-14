from pathlib import Path

from agentlego.parsers import NaiveParser
from agentlego.testing import setup_tool
from agentlego.tools import (AudioImageToImage, AudioTextToImage, AudioToImage,
                             ThermalToImage)
from agentlego.types import ImageIO

data_dir = Path(__file__).parents[2] / 'data'


def test_audio_to_image():
    tool = setup_tool(AudioToImage, device='cuda')
    tool.set_parser(NaiveParser)
    res = tool(data_dir / 'audio/cat.wav')
    assert isinstance(res, ImageIO)


def test_thermal_to_image():
    tool = setup_tool(ThermalToImage, device='cuda')
    tool.set_parser(NaiveParser)
    res = tool(data_dir / 'audio/030419.jpg')
    assert isinstance(res, ImageIO)


def test_audio_image_to_image():
    tool = setup_tool(AudioImageToImage, device='cuda')
    tool.set_parser(NaiveParser)
    res = tool(data_dir / 'images/dog.jpg', data_dir / 'audio/cat.wav')
    assert isinstance(res, ImageIO)


def test_audio_text_to_image():
    tool = setup_tool(AudioTextToImage, device='cuda')
    tool.set_parser(NaiveParser)
    res = tool(data_dir / 'audio/cat.wav', 'generate a cat flying in the sky')
    assert isinstance(res, ImageIO)
