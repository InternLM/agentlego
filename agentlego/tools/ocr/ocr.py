from typing import Callable, Sequence, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class OCR(BaseTool):
    """A tool to recognize the optical characters on an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        lang (str | Sequence[str]): The language to be recognized.
            Defaults to 'en'.
        device (str | bool): The device to load the model. Defaults to True,
            which means automatically select device.
        **read_args: Other keyword arguments for read text. Please check the
            `EasyOCR docs <https://www.jaided.ai/easyocr/documentation/>`_.
    """
    DEFAULT_TOOLMETA = ToolMeta(
        name='OCR',
        description='This tool can recognize all text on the input image.',
        inputs=['image'],
        outputs=['text'],
    )

    @require('easyocr')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 lang: Union[str, Sequence[str]] = 'en',
                 device: Union[bool, str] = True,
                 line_group_tolerance = -1,
                 **read_args):
        super().__init__(toolmeta=toolmeta, parser=parser)
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

    def apply(self, image: ImageIO) -> str:

        image = image.to_array()
        if self.line_group_tolerance >= 0:
            results = self._reader.readtext(image, **self.read_args)
            results.sort(key=lambda x: x[0][0][1])

            lines = []
            line = [results[0]]

            for result in results[1:]:
                if abs(result[0][0][1] - line[0][0][0][1]) <= self.line_group_tolerance:
                    line.append(result)
                else:
                    lines.append(line)
                    line = [result]

            lines.append(line)

            ocr_results = []
            for line in lines:
                # For each line, sort the elements by their left x-coordinate and join their texts
                sorted_line = sorted(line, key=lambda x: x[0][0][0])
                text_line = ' '.join(item[1] for item in sorted_line)
                ocr_results.append(text_line)
        else:
            ocr_results = self._reader.readtext(image, detail=0, **self.read_args)
        outputs = '\n'.join(ocr_results)
        return outputs
