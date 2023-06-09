import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('mmdet'), reason='requires mmdet')
class TestText2BoxTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('Text2BoxTool', device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, man')
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'Text2BoxTool', output_style='pil image', device='cuda')
        res = tool(f'{img_path}, man')
        assert isinstance(res, Image.Image)


@skipIf(not is_installed('mmdet'), reason='requires mmdet')
class TestObjectDetectionTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('ObjectDetectionTool', device='cuda')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        tool = load_tool(
            'ObjectDetectionTool',
            input_style='pil image',
            output_style='pil image',
            device='cuda')
        img = Image.fromarray(img)
        res = tool(img)
        assert isinstance(res, Image.Image)
