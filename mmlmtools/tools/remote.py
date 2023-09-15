# Copyright (c) OpenMMLab. All rights reserved.
import base64
from io import BytesIO
from typing import List, Optional, Union
from urllib.parse import urljoin

import requests

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.tools.base import BaseTool
from mmlmtools.types import AudioIO, ImageIO


class RemoteTool(BaseTool):

    def __init__(
        self,
        url,
        toolmeta: Union[dict, ToolMeta, None] = None,
        input_fields: Optional[List[str]] = None,
        parser=DefaultParser,
    ):
        if not url.endswith('/'):
            url += '/'
        self.url = url

        if toolmeta is None or input_fields is None:
            toolmeta, input_fields = self.request_meta()

        self._input_fields = input_fields
        super().__init__(toolmeta, parser)

    def request_meta(self):
        url = urljoin(self.url, 'meta')
        response = requests.get(url).json()
        return response['toolmeta'], response['input_fields']

    def apply(self, *args):
        form = {}
        for input_field, arg in zip(self.input_fields, args):
            if isinstance(arg, str):
                form[input_field] = (None, arg)
            elif isinstance(arg, (ImageIO, AudioIO)):
                file = arg.to_path()
                form[input_field] = (file, open(file, 'rb'))
            else:
                raise NotImplementedError()

        url = urljoin(self.url, 'call')
        try:
            response = requests.post(url, files=form).json()
        except requests.ConnectionError as e:
            raise ConnectionError(
                f'Failed to connect the remote tool `{self.name}`.') from e
        except requests.JSONDecodeError:
            raise RuntimeError('Unexcepted server response.')

        if isinstance(response, dict):
            if 'error' in response:
                # Tool internel error
                raise RuntimeError(response['error'])
            elif 'detail' in response:
                # FastAPI validation error
                msg = response['detail']['msg']
                err_type = response['detail']['type']
                raise ValueError(f'{err_type}({msg})')

        parsed_res = []
        for res in response:
            if not isinstance(res, dict):
                data = res
            elif res['type'] == 'image':
                from PIL import Image
                file = BytesIO(base64.decodebytes(res['data'].encode('ascii')))
                data = ImageIO(Image.open(file))
            elif res['type'] == 'audio':
                import torchaudio
                file = BytesIO(base64.decodebytes(res['data'].encode('ascii')))
                data = AudioIO(*torchaudio.load(file, format='wav'))
            parsed_res.append(data)

        return parsed_res[0] if len(parsed_res) == 1 else tuple(parsed_res)

    @classmethod
    def from_server(cls, url: str) -> List['RemoteTool']:
        response = requests.get(url).json()
        tools = []
        for tool_info in response:
            tool = cls(
                url=urljoin(url, tool_info['domain'] + '/'),
                toolmeta=tool_info['toolmeta'],
                input_fields=tool_info['input_fields'],
            )
            tools.append(tool)
        return tools

    @property
    def input_fields(self):
        return self._input_fields
