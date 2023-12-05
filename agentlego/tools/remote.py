# Copyright (c) OpenMMLab. All rights reserved.
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from agentlego.parsers import DefaultParser
from agentlego.schema import Parameter, ToolMeta
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, ImageIO
from agentlego.utils import temp_path


class RemoteTool(BaseTool):

    def __init__(
        self,
        url,
        toolmeta: Union[dict, ToolMeta, None] = None,
        parameters: Optional[Dict[str, Parameter]] = None,
        parser=DefaultParser,
    ):
        if not url.endswith('/'):
            url += '/'
        self.url = url

        if toolmeta is None or parameters is None:
            toolmeta, parameters = self.request_meta()

        self._parameters = parameters
        super().__init__(toolmeta, parser)

    def request_meta(self):
        url = urljoin(self.url, 'meta')
        response = requests.get(url).json()
        toolmeta = response['toolmeta']
        parameters = {
            p['name']: Parameter(**p)
            for p in response['parameters']
        }
        return toolmeta, parameters

    def apply(self, *args, **kwargs):
        for arg, arg_name in zip(args, self.parameters):
            kwargs[arg_name] = arg

        form = {}
        for k, v in kwargs.items():
            if isinstance(v, (ImageIO, AudioIO)):
                file = v.to_path()
                form[k] = (file, open(file, 'rb'))
            else:
                form[k] = (None, v)

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
                # Tool internal error
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
                filename = temp_path('audio', '.wav')
                with open(filename, 'wb') as f:
                    f.write(base64.decodebytes(res['data'].encode('ascii')))
                data = AudioIO(filename)
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
                parameters={
                    p['name']: Parameter(**p)
                    for p in tool_info['parameters']
                },
            )
            tools.append(tool)
        return tools

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self._parameters
