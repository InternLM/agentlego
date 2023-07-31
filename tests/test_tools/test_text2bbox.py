import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import HuggingFaceAgentParser, VisualChatGPTParser


@skipIf(not is_installed('mmdet'), reason='requires mmdet')
class TestText2BboxTool(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'Text2BboxTool', parser=VisualChatGPTParser(), device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, man')
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'Text2BboxTool', parser=HuggingFaceAgentParser(), device='cuda')
        res = tool(img, 'man')
        assert isinstance(res, Image.Image)
