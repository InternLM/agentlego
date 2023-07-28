import os.path as osp
from unittest import skipIf

import cv2
import numpy as np
from mmengine import is_installed
from PIL import Image

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestText2ImageTool(ToolTestCase):

    def test_call(self):
        # tool = load_tool('Text2ImageTool', device='cuda')
        tool = load_tool('Text2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        # tool = load_tool(
        #     'Text2ImageTool', output_style='pil image', device='cuda')
        tool = load_tool(
            'Text2ImageTool', output_style='pil image', device='cpu')
        res = tool(f'{img_path}')
        assert isinstance(res, Image.Image)


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestCanny2ImageTool(ToolTestCase):

    def test_call(self):
        # tool = load_tool('Canny2ImageTool', device='cuda')
        tool = load_tool('Canny2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, prompt')
        assert isinstance(res, str)


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestSeg2ImageTool(ToolTestCase):

    def test_all(self):
        # tool = load_tool('Seg2ImageTool', device='cuda')
        tool = load_tool('Seg2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, prompt')
        assert isinstance(res, str)


@skipIf(not is_installed('mmagic'), reason='requires mmagic')
class TestPose2ImageTool(ToolTestCase):

    def test_all(self):
        # tool = load_tool('Pose2ImageTool', device='cuda')
        tool = load_tool('Pose2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, prompt')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestScribbleText2ImageTool(ToolTestCase):

    def test_all(self):
        # tool = load_tool('ScribbleText2ImageTool', device='cuda')
        tool = load_tool('ScribbleText2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, prompt')
        assert isinstance(res, str)


@skipIf(not is_installed('diffusers'), reason='requires diffusers')
class TestDepthText2ImageTool(ToolTestCase):

    def test_all(self):
        # tool = load_tool('DepthText2ImageTool', device='cuda')
        tool = load_tool('DepthText2ImageTool', device='cpu')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, prompt')
        assert isinstance(res, str)
