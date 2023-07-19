# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmocr.apis import MMOCRInferencer
from PIL import Image

from mmlmtools.cached_dict import CACHED_TOOLS
from mmlmtools.toolmeta import ToolMeta
from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class OCRTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Recognize the Optical Characters On Image',
        model={
            'det': 'dbnetpp',
            'rec': 'svtr-small'
        },
        description='This is a useful tool '
        'when you want to recognize the text from a photo.',
        input_description='It takes a string as the input, '
        'representing the image_path. ',
        output_description='It returns a string as the output, '
        'representing the characters or words in the image. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            if CACHED_TOOLS.get('mmocr_inferencer', None) is not None:
                self._inferecner = CACHED_TOOLS['mmocr_inferencer']
            else:
                self._inferecner = MMOCRInferencer(
                    det=self.toolmeta.model['det'],
                    rec=self.toolmeta.model['rec'],
                    device=self.device)
                CACHED_TOOLS['mmocr_inferencer'] = self._inferecner

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_name(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            ocr_results = self._inferencer(
                inputs, show=False)['predictions'][0]
            outputs = ocr_results['rec_texts']
        return outputs

    def convert_outputs(self, outputs):
        if self.output_style == 'text':
            outputs = '\n'.join(outputs)
            return outputs
        else:
            raise NotImplementedError


class ImageMaskOCRTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Recognize The Optical Characters On Image With Mask',
        model=None,
        description='This is a useful tool '
        'when you want to  recognize the characters or words in the masked '
        'region of the image. '
        'like: recognize the characters or words in the masked region. ',
        input_description='The input to this tool should be a comma separated'
        ' string of two, representing the image_path and mask_path. ',
        output_description='It returns a string as the output, '
        'representing the characters or words in the image. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, audio_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device,
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            if CACHED_TOOLS.get('mmocr_inferencer', None) is not None:
                self._inferecner = CACHED_TOOLS['mmocr_inferencer']
            else:
                self._inferecner = MMOCRInferencer(
                    det=self.toolmeta.model['det'],
                    rec=self.toolmeta.model['rec'],
                    device=self.device)
                CACHED_TOOLS['mmocr_inferencer'] = self._inferecner

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, mask_path':  # visual chatgpt style  # noqa
            image_path, mask_path = inputs.split(',')
            image_path, mask_path = image_path.strip(), mask_path.strip()
            return image_path, mask_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        image_path, mask_path = inputs
        if self.remote:
            raise NotImplementedError
        else:
            image_path = image_path.strip()
            mask_path = mask_path.strip()
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.uint8)
            ocr_results = self._inferencer(inputs, show=False)
            ocr_results = ocr_results['predictions'][0]
            seleted_ocr_text = self.get_ocr_by_mask(mask, ocr_results)

        return seleted_ocr_text

    def get_ocr_by_mask(self, mask, ocr_res):
        inds = np.where(mask != 0)
        inds = (inds[0][::8], inds[1][::8])
        # self.result = self.reader.readtext(self.image_path)
        if len(inds[0]) == 0:
            # self.result = self.reader.readtext(image_path)
            return 'No characters in the image'

        ocr_text_list = []
        num_mask = len(inds[0])
        for i in range(num_mask):
            res = self.search((inds[1][i], inds[0][i]), ocr_res)
            if res is not None and len(res) > 0:
                ocr_text_list.append(res)
        ocr_text_list = list(dict.fromkeys(ocr_text_list))  # remove duplicates
        ocr_text = '\n'.join(ocr_text_list)
        if ocr_text is None or len(ocr_text.strip()) == 0:
            ocr_text = 'No characters in the image'
        else:
            ocr_text = '\n' + ocr_text

        return ocr_text

    def search(self, coord, ocr_res):
        det_bboxes = ocr_res['det_bboxes']
        rec_texts = ocr_res['rec_texts']
        for i, item in det_bboxes:
            left_top = item[:2]
            right_bottom = item[2:]
            if (coord[0] >= left_top[0] and coord[1] >= left_top[1]) and \
               (coord[0] <= right_bottom[0] and coord[1] <= right_bottom[1]):
                return rec_texts[i]

        return ''

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
