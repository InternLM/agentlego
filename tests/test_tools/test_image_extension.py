import os.path as osp

import numpy as np
import cv2
from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase
from PIL import Image


class TestImageExtensionTool(ToolTestCase):

    def test_call(self):
        tool = load_tool('ImageExtensionTool', device='cpu')
        img = Image.open('tests/data/images/test-image.jpeg')
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img = np.array(img)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(f'{img_path}, 800x800')
        assert isinstance(res, str)
