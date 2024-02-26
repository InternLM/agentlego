import json
import os

import requests

from agentlego.types import ImageIO
from ..base import BaseTool


class MathOCR(BaseTool):
    """A tool to recognize math expressions from an image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can recognize math expressions from an '
                    'image and return the latex style expression.')

    def __init__(self, app_id: str = 'ENV', app_key: str = 'ENV', toolmeta=None):
        super().__init__(toolmeta=toolmeta)

        if app_id == 'ENV':
            app_id = os.getenv('MATHPIX_APP_ID', None)
        if app_key == 'ENV':
            app_key = os.getenv('MATHPIX_APP_KEY', None)
        if not app_key or not app_id:
            raise ValueError('Please set Mathpix app id and key or use `MATHPIX_APP_ID` '
                             'and `MATHPIX_APP_KEY` environment variables.')
        self.app_id = app_id
        self.app_key = app_key

    def apply(self, image: ImageIO) -> str:
        exception = RuntimeError('Failed to recognize math expression.')
        for _ in range(3):
            try:
                response = requests.post(
                    'https://api.mathpix.com/v3/text',
                    files={'file': open(image.to_path(), 'rb')},
                    data={
                        'options_json':
                        json.dumps({
                            'math_inline_delimiters': ['$', '$'],
                            'rm_spaces': True
                        })
                    },
                    headers={
                        'app_id': self.app_id,
                        'app_key': self.app_key,
                    },
                    timeout=60,
                )
                return response.text
            except Exception as e:
                exception = e
                continue
        raise exception
