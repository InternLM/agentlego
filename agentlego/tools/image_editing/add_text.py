from PIL import ImageDraw, ImageFont
import os
import re
from agentlego.types import ImageIO
from ..base import BaseTool


class AddText(BaseTool):
    """A tool to draw a box on a certain region of the input image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    def __init__(self, toolmeta=None):
        super().__init__(toolmeta=toolmeta)

    def apply(self, image: ImageIO, text: str, position: str, 
              font_size: int = 40, 
              font_file_name: str = 'Roboto-Medium.ttf',
              color: str = 'red') -> str:
        """ 
        position: 
        (x, y) x, y >= 1, absolute position, lower left corner of the text
        or
        (x, y) 0 < x, y < 1, relative position, in the proportion of image width and height
        or
        upper left | upper | upper right
        left | middle | right
        lower left | lower | lower right
        """
        image = image.to_pil()
        draw = ImageDraw.Draw(image)
        current_file_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(current_file_path)
        font_path = os.path.abspath(os.path.join(cur_dir, 'fonts', font_file_name))
        font = ImageFont.truetype(font_path, font_size)
        tw = draw.textlength(text, font=font)
        th = font_size
        w, h = image.size
        m = w//20  # margin
        POS = {
        'upper left': (m, h//6-th//2),
        'upper': (w//2-tw//2, h//6-th//2),
        'upper right': (w-m-tw, h//6-th//2),
        'left': (m, h//2-th//2),
        'middle': (w//2-tw//2, h//2-th//2),
        'right': (w-m-tw, h//2-th//2),
        'lower left': (m, 5*h//6-th//2),
        'lower': (w//2-tw//2, 5*h//6-th//2),
        'lower right': (w-m-tw, 5*h//6-th//2)
        }
        if bool(re.match(r"\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)", position.replace(" ", ""))):
            pos = eval(position.replace(" ", ""))
            if pos[0] > 0 and pos[0] < 1 and pos[1] > 0 and pos[1] < 1:
                pos = (int(pos[0]*w), int(pos[1]*h))
            elif pos[0] >= 1 and pos[1] >= 1:
                pass
            else:
                raise ValueError('position tuple (x,y) is illegal.')
        elif position in POS.keys():
            pos = POS[position]
        else:
            raise ValueError(f'position should be a tuple (x,y), or a string in {str(POS.keys())}.')
        draw.text(pos, text, fill=color, font=font)
        del draw
        return ImageIO(image)
