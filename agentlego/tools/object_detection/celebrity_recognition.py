from ..base import BaseTool
import json
import boto3


class CelebrityRecognition(BaseTool):
    """A tool to recognize celebrities in an image.

    Args:
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can recognize celebrities in an image.')

    def __init__(self, toolmeta=None):
        super().__init__(toolmeta=toolmeta)

    def apply(self, path: str) -> dict:
        for _ in range(3):
            try:
                session = boto3.Session(profile_name='profile-name')
                client = session.client('rekognition')
                with open(path, 'rb') as image:
                    r = client.recognize_celebrities(Image={'Bytes': image.read()})
                return r
            except Exception as e:
                print(f'{type(e)}: {str(e)}')
                time.sleep(1)
                continue
