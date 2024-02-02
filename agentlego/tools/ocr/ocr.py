from typing import Sequence, Tuple, Union

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class OCR(BaseTool):
    """A tool to recognize the optical characters on an image.

    Args:
        lang (str | Sequence[str]): The language to be recognized.
            Defaults to 'en'.
        line_group_tolerance (int): The line group tolerance threshold.
            Defaults to -1, which means to disable the line group method.
        device (str | bool): The device to load the model. Defaults to True,
            which means automatically select device.
        **read_args: Other keyword arguments for read text. Please check the
            `EasyOCR docs <https://www.jaided.ai/easyocr/documentation/>`_.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can recognize all text on the input image.'

    @require('easyocr')
    def __init__(self,
                 lang: Union[str, Sequence[str]] = 'en',
                 line_group_tolerance: int = -1,
                 device: Union[bool, str] = True,
                 toolmeta=None,
                 **read_args):
        super().__init__(toolmeta=toolmeta)
        if isinstance(lang, str):
            lang = [lang]
        self.lang = list(lang)
        self.read_args = read_args
        self.device = device
        self.line_group_tolerance = line_group_tolerance
        read_args.setdefault('decoder', 'beamsearch')

        if line_group_tolerance >= 0:
            read_args.setdefault('paragraph', False)
        else:
            read_args.setdefault('paragraph', True)

    def setup(self):
        import easyocr
        self._reader: easyocr.Reader = load_or_build_object(
            easyocr.Reader, self.lang, gpu=self.device)

    def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('OCR results, include bbox in x1, y1, x2, y2 format '
                        'and the recognized text.')]:

        image = image.to_array()
        results = self._reader.readtext(image, detail=1, **self.read_args)
        results = [(self.extract_bbox(item[0]), item[1]) for item in results]

        if self.line_group_tolerance >= 0:
            results.sort(key=lambda x: x[0][1])

            groups = []
            group = []

            for item in results:
                if not group:
                    group.append(item)
                    continue

                if abs(item[0][1] - group[-1][0][1]) <= self.line_group_tolerance:
                    group.append(item)
                else:
                    groups.append(group)
                    group = [item]

            groups.append(group)

            results = []
            for group in groups:
                # For each line, sort the elements by their left x-coordinate and join their texts
                line = sorted(group, key=lambda x: x[0][0])
                bboxes = [item[0] for item in line]
                text = ' '.join(item[1] for item in line)
                results.append((self.extract_bbox(bboxes), text))

        outputs = []
        for item in results:
            outputs.append('({}, {}, {}, {}) {}'.format(*item[0], item[1]))
        outputs = '\n'.join(outputs)
        return outputs

    @staticmethod
    def extract_bbox(char_boxes) -> Tuple[int, int, int, int]:
        xs = [int(box[0]) for box in char_boxes]
        ys = [int(box[1]) for box in char_boxes]
        return min(xs), min(ys), max(xs), max(ys)
