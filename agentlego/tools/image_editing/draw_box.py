from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import parse_multi_float
from ..base import BaseTool


class DrawBox(BaseTool):
    default_desc = 'A tool to draw a box on a certain region of the input image.'

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        annotation: Annotated[Optional[str],
                              Info('The extra annotation text of the bbox')] = None,
    ) -> ImageIO:
        image_pil = image.to_pil().copy().convert('RGBA')
        canvas = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        # Estimate a proper font size.
        fontsize = int(image_pil.size[1] / 336 * 18)
        font = ImageFont.load_default(size=fontsize)

        x1, y1, x2, y2 = parse_multi_float(bbox, 4)

        draw.rectangle(
            (x1, y1, x2, y2),
            fill=(255, 0, 0, 64),
            outline=(255, 0, 0, 255),
        )
        if annotation:
            draw.text(
                (x1, y1 - 5),
                annotation,
                fill=(255, 0, 0, 255),
                anchor='lb',
                font=font,
            )
        return ImageIO(Image.alpha_composite(image_pil, canvas).convert('RGB'))
