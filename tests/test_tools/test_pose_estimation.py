import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('mmpose'), reason='requires mmpose')
class TestHumanBodyPoseTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('HumanBodyPoseTool', device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'HumanBodyPoseTool',
            output_style='pil image',
            input_style='pil image',
            device='cuda')
        res = tool(img)
        assert isinstance(res, Image.Image)


@skipIf(not is_installed('mmpose'), reason='requires mmpose')
class TestHumanFaceLandmarkTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('HumanFaceLandmarkTool', device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'HumanBodyPoseTool',
            output_style='pil image',
            input_style='pil image',
            device='cuda')
        res = tool(img)
        assert isinstance(res, Image.Image)
