import os.path as osp

import cv2
import numpy as np
from PIL import Image

from mmlmtools.api import load_tool
from mmlmtools.testing import ToolTestCase


class TestImageEdge(ToolTestCase):

    def test_call(self):
        tool = load_tool('Image2CannyTool')
        img = np.ones([224, 224, 3]).astype(np.uint8)
        img_path = osp.join(self.tempdir.name, 'temp.jpg')
        cv2.imwrite(img_path, img)
        res = tool(img_path)
        assert isinstance(res, str)

        img = Image.fromarray(img)
        tool = load_tool(
            'Image2CannyTool',
            input_style='pil image',
            output_style='pil image')
        res = tool(img)
        assert isinstance(res, Image.Image)