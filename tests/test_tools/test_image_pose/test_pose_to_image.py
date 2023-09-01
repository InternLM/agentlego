import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, LangChainParser


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestPoseToImage(ToolTestCase):

    def test_all(self):
        tool = load_tool(
            'PoseToImage', parser=LangChainParser(), device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path, 'prompt')
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'PoseToImage', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool(img, 'prompt')
        assert isinstance(res, Image.Image)
