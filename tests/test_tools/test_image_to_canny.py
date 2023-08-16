import os.path as osp

import cv2
import numpy as np
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


class TestImageToCanny(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'ImageToCanny', parser=VisualChatGPTParser(), device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'ImageToCanny', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool(img)
        assert isinstance(res, Image.Image)
