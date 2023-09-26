import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from agentlego import load_tool
from agentlego.testing import ToolTestCase
from agentlego.tools.parsers import HuggingFaceAgentParser, LangChainParser


@skipIf(not is_installed('mmpose'), reason='requires mmpose')
class TestHumanBodyPose(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'HumanBodyPose', parser=LangChainParser(), device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'HumanBodyPose', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool(img)
        assert isinstance(res, Image.Image)
