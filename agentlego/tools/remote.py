import base64
import re
from io import BytesIO, IOBase
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit

import requests

from agentlego.parsers import DefaultParser
from agentlego.schema import Parameter, ToolMeta
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, ImageIO
from agentlego.utils import temp_path
from agentlego.utils.openapi import (PRIMITIVE_TYPES, APIOperation,
                                     APIPropertyBase, APIResponseProperty,
                                     OpenAPISpec)


def image_to_byte_file(image: ImageIO) -> IOBase:
    file = BytesIO()
    image.to_pil().save(file, 'PNG')
    file.seek(0)
    return file


def audio_to_byte_file(audio: AudioIO) -> IOBase:
    try:
        import torchaudio
        file = BytesIO()
        torchaudio.save(file, audio.to_tensor(), audio.sampling_rate)
        file.seek(0)
        return file
    except ImportError:
        return open(audio.to_path(), 'rb')


def base64_to_image(encoded: str) -> ImageIO:
    from PIL import Image
    file = BytesIO(base64.b64decode(encoded))
    return ImageIO(Image.open(file))


def base64_to_audio(encoded: str) -> AudioIO:
    try:
        import torchaudio
        file = BytesIO(base64.b64decode(encoded))
        audio, sr = torchaudio.load(file)
        return AudioIO(audio, sampling_rate=sr)
    except ImportError:
        filename = temp_path('audio', '.wav')
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(encoded))
        return AudioIO(filename)


class RemoteTool(BaseTool):

    def __init__(
        self,
        url,
        operation: Optional[APIOperation] = None,
        method: str = 'post',
        headers: Optional[dict] = None,
        auth=None,
        parser=DefaultParser,
    ):
        self.url = url
        self.headers = headers
        self.auth = auth

        if operation is None:
            operation = APIOperation.from_openapi_url(
                # The default openapi file for the tool server.
                spec_url=urljoin(url, '/openapi.json'),
                path=urlsplit(url).path,
                method=method,
            )
        self.operation = operation
        self.toolmeta = self._get_toolmeta(operation)
        self.set_parser(parser)
        self._is_setup = False

    def _construct_path(self, kwargs: Dict[str, str]) -> str:
        """Construct the path from the tool input."""
        path = self.url
        for param in self.operation.path_params:
            path = path.replace(f'{{{param}}}', str(kwargs.pop(param, '')))
        return path

    def _extract_query_params(self, kwargs: Dict[str, str]) -> Dict[str, str]:
        """Extract the query params from the tool input."""
        query_params = {}
        for param in self.operation.query_params:
            if param in kwargs:
                query_params[param] = kwargs.pop(param)
        return query_params

    def _extract_body_params(self,
                             kwargs: Dict[str,
                                          str]) -> Optional[Dict[str, str]]:
        """Extract the request body params from the tool input."""
        body_params = None
        if self.operation.body_params:
            body_params = {}
            for param in self.operation.body_params:
                if param in kwargs:
                    value = kwargs.pop(param)
                    if isinstance(value, ImageIO):
                        value = image_to_byte_file(value)
                    elif isinstance(value, AudioIO):
                        value = audio_to_byte_file(value)
                    body_params[param] = value
        return body_params

    def apply(self, *args, **kwargs):
        for arg, p in zip(args, self.inputs):
            kwargs[p.name] = arg

        request_args = {
            'url': self._construct_path(kwargs),
            'params': self._extract_query_params(kwargs)
        }
        if self.operation.request_body:
            media_type = self.operation.request_body.media_type
            body = self._extract_body_params(kwargs)
            if media_type == 'multipart/form-data':
                request_args['files'] = {
                    k: (k, v) if isinstance(v, IOBase) else (None, v)
                    for k, v in body.items()
                }
            elif media_type == 'application/json':
                request_args['json'] = body
            else:
                assert media_type == 'application/x-www-form-urlencoded'
                request_args['data'] = body

        method = getattr(requests, self.operation.method.value)
        try:
            response: requests.Response = method(
                **request_args,
                headers=self.headers,
                auth=self.auth,
            )
        except requests.ConnectionError as e:
            raise ConnectionError(
                f'Failed to connect the remote tool `{self.name}`.') from e
        if response.status_code != 200:
            if response.headers.get('Content-Type') == 'application/json':
                content = response.json()
            else:
                content = response.content.decode()
            raise RuntimeError(
                f'Failed to call the remote tool `{self.name}` '
                f'because of {response.reason}.\nResponse: {content}')
        try:
            response = response.json()
        except requests.JSONDecodeError:
            raise RuntimeError(f'Failed to call the remote tool `{self.name}` '
                               'because of unknown response.\n'
                               f'Response: {response.content.decode()}')

        response_schema = self.operation.responses
        if response_schema is None or response_schema.get('200') is None:
            # Directly use string if the response schema is not specified
            return str(response)
        else:
            out_props = response_schema['200'].properties

        if isinstance(out_props, APIResponseProperty):
            return self._parse_output(response, out_props)
        elif isinstance(out_props, list):
            return tuple(
                self._parse_output(out, p)
                for out, p in zip(response, out_props))
        else:
            return {
                p.name: self._parse_output(out, p)
                for out, p in zip(response, out_props)
            }

    @staticmethod
    def _parse_output(out, p: APIResponseProperty):
        fmt_pattern = r'(\w+)(/(\w+))?(;(\w+))?'
        schema_format = re.match(fmt_pattern, p.format or '')
        if schema_format:
            media_type, _, _, _, encoding = schema_format.groups()
        else:
            media_type, encoding = 'unknown', None

        if media_type == 'image' and encoding == 'base64':
            out = base64_to_image(out)
        elif media_type == 'audio' and encoding == 'base64':
            out = base64_to_audio(out)
        else:
            out = out
        return out

    @classmethod
    def from_server(cls, url: str, **kwargs) -> List['RemoteTool']:
        return cls.from_openapi(url=urljoin(url, '/openapi.json'), **kwargs)

    @classmethod
    def from_openapi(
        cls,
        url: str,
        toolkit: Union[str, bool] = False,
        **kwargs,
    ) -> List['RemoteTool']:
        if url.startswith('http'):
            spec = OpenAPISpec.from_url(url)
        else:
            spec = OpenAPISpec.from_file(url)

        if isinstance(toolkit, bool):
            toolkit = spec.info.title.replace(' ', '_') if toolkit else None

        tools = []
        for path, method in spec.iter_all_method():
            operation = APIOperation.from_openapi_spec(spec, path, method)
            tool = cls(
                url=urljoin(operation.base_url, operation.path),
                operation=operation,
                **kwargs,
            )
            tools.append(tool)
        return tools

    @staticmethod
    def _get_toolmeta(
        operation: APIOperation,
        toolkit: Optional[str] = None,
    ) -> Tuple[ToolMeta, List[Parameter]]:
        name = operation.operation_id
        if toolkit is not None:
            name = toolkit + '.' + name
        inputs = RemoteTool._get_inputs(operation)
        outputs = RemoteTool._get_outputs(operation)
        toolmeta = ToolMeta(
            name=name,
            description=operation.description,
            inputs=inputs,
            outputs=outputs,
        )

        return toolmeta

    @staticmethod
    def _get_inputs(op: APIOperation) -> List[Parameter]:
        inputs = []
        properties = []
        if op.properties:
            properties.extend(op.properties)
        if op.request_body and op.request_body.properties:
            properties.extend(op.request_body.properties)
        for p in properties:
            inputs.append(RemoteTool._prop_to_parameter(p))
        return inputs

    @staticmethod
    def _get_outputs(op: APIOperation) -> List[Parameter]:
        if op.responses is None or op.responses.get('200') is None:
            # If not specify outputs, directly handle as a single text.
            outputs = [Parameter(type=str)]

        out_props = op.responses['200'].properties
        if isinstance(out_props, list):
            outputs = [RemoteTool._prop_to_parameter(out) for out in out_props]
        elif isinstance(out_props, dict):
            outputs = [
                RemoteTool._prop_to_parameter(out)
                for out in out_props.values()
            ]
        else:
            outputs = [RemoteTool._prop_to_parameter(out_props)]

        return outputs

    @staticmethod
    def _prop_to_parameter(prop: APIPropertyBase) -> Parameter:
        p_type = PRIMITIVE_TYPES.get(prop.type)
        if p_type is str:
            schema_format = prop.format or ''
            if 'image' in schema_format:
                p_type = ImageIO
            elif 'audio' in schema_format:
                p_type = AudioIO
        return Parameter(
            type=p_type,
            name=prop.name if prop.name != '_null' else None,
            description=prop.description,
            optional=not prop.required,
            default=prop.default,
        )
