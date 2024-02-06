from ..base import BaseTool
import requests
import json


class MathOCR(BaseTool):
    """A tool to recognize math expressions from an image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can recognize math expressions from an image.')

    def __init__(self, toolmeta=None):
        super().__init__(toolmeta=toolmeta)

    def apply(self, path: str) -> str:
        for _ in range(3):
            try:
                r = requests.post("https://api.mathpix.com/v3/text",
                    files={"file": open(path,"rb")},
                    data={
                        "options_json": json.dumps({
                            "math_inline_delimiters": ["$", "$"],
                            "rm_spaces": True
                            })
                        },
                    headers={
                        "app_id": "APP_ID",
                        "app_key": "APP_KEY"
                    },
                    timeout=60
                )
                return r["text"]
            except Exception as e:
                print(f'{type(e)}: {str(e)}')
                time.sleep(1)
                continue
