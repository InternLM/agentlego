from PIL import ImageDraw
from agentlego.types import ImageIO
from ..base import BaseTool


class DrawBox(BaseTool):
    """A tool to draw a box on a certain region of the input image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    def __init__(self, toolmeta=None):
        super().__init__(toolmeta=toolmeta)

    def apply(self, image: ImageIO, region: str, 
                                outline: str = 'red', width: int = 5) -> str:
        """ 
        region: (x1, y1, x2, y2)  
        x1, y1: upper left
        x2, y2: lower right
        """
        image = image.to_pil()
        draw = ImageDraw.Draw(image)
        draw.rectangle(eval(region),outline=outline,width=width)
        del draw
        return ImageIO(image)
