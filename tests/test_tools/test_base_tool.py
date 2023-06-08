import os.path as osp
import tempfile
from unittest import TestCase

import cv2
import numpy as np
from PIL import Image

from mmlmtools.tools.base_tool import BaseTool


class ResizeTool(BaseTool):
    DEFAULT_TOOLMETA = dict(description='resize tool', tool_name='ResizeTool')

    def setup(self):
        ...

    def apply(self, inputs, size=(224, 224)):
        outputs = cv2.imread(inputs)
        outputs = cv2.resize(outputs, size)
        ret_path = f'{inputs}_results.jpg'
        cv2.imwrite(ret_path, outputs)
        return ret_path


class TestBaseTool(TestCase):

    def test_call(self):
        with tempfile.TemporaryDirectory() as tempdir:
            img = np.ones((20, 20, 3)).astype(np.uint8)
            img_path = osp.join(tempdir, 'img.jpg')
            cv2.imwrite(img_path, img)

            # pil image IN -> pil image OUT
            tool = ResizeTool(
                input_style='pil image', output_style='pil image')
            img = Image.fromarray(img)
            result = tool(img)
            assert result.size == (224, 224)

            # f"{image_path}, {size}" IN -> image_path OUT
            tool = ResizeTool(input_style='{image_path}, {eval}')
            result = tool(f'{img_path}, (224, 224)')
            assert cv2.imread(result).shape == (224, 224, 3)

            # f"{image_path}, {size}" IN -> pil image OUT
            tool = ResizeTool(
                input_style='{image_path}, {eval}', output_style='pil image')
            result = tool(f'{img_path}, (224, 224)')
            assert result.size == (224, 224)
