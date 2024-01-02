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
from agentlego.utils import APIOperation, OpenAPISpec, temp_path
from agentlego.utils.openapi.api_model import (APIPropertyBase,
                                               APIResponseProperty)


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
        toolmeta, parameters = self._get_toolmeta_and_params(operation)
        self._parameters = parameters
        super().__init__(toolmeta, parser)

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
        for arg, arg_name in zip(args, self.parameters):
            kwargs[arg_name] = arg

        request_args = {
            'url': self._construct_path(kwargs),
            'params': self._extract_query_params(kwargs)
        }
        if self.operation.request_body:
            if self.operation.request_body.media_type == 'multipart/form-data':
                request_args['files'] = self._extract_body_params(kwargs)
            else:
                request_args['data'] = self._extract_body_params(kwargs)

        method = getattr(requests, self.operation.method.value)
        try:
            requests.post
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

        out_props = self._get_outputs(self.operation)
        if out_props is None:
            # Directly use string if the response schema is not specified
            return str(response)
        elif isinstance(out_props, APIResponseProperty):
            return self._parse_output(response)
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
    def _get_toolmeta_and_params(
        operation: APIOperation,
        toolkit: Optional[str] = None,
    ) -> Tuple[ToolMeta, List[Parameter]]:
        name = operation.operation_id
        if toolkit is not None:
            name = toolkit + '.' + name
        parameters = RemoteTool._get_parameters(operation)
        out_props = RemoteTool._get_outputs(operation)
        if out_props is None:
            # If not specify outputs, directly handle as a single text.
            outputs = ['str']
        elif isinstance(out_props, list):
            outputs = [RemoteTool._extract_category(out) for out in out_props]
        elif isinstance(out_props, dict):
            out_props = out_props.values()
            outputs = [RemoteTool._extract_category(out) for out in out_props]
        else:
            outputs = [RemoteTool._extract_category(out_props)]

        toolmeta = ToolMeta(
            name=name,
            description=operation.description,
            inputs=[p.category for p in parameters.values()],
            outputs=outputs,
        )

        return toolmeta, parameters

    @staticmethod
    def _get_parameters(op: APIOperation) -> Dict[str, Parameter]:
        params = {}
        properties = []
        if op.properties:
            properties.extend(op.properties)
        if op.request_body and op.request_body.properties:
            properties.extend(op.request_body.properties)
        for p in properties:
            params[p.name] = Parameter(
                name=p.name,
                category=RemoteTool._extract_category(p),
                description=p.description,
                default=p.default,
                optional=not p.required,
            )
        return params

    @staticmethod
    def _get_outputs(op: APIOperation):
        if op.responses is None or op.responses.get('200') is None:
            return None

        return op.responses['200'].properties

    @staticmethod
    def _extract_category(param: APIPropertyBase) -> str:
        if param.type == 'string':
            schema_format = param.format or ''
            if 'image' in schema_format:
                return 'image'
            elif 'audio' in schema_format:
                return 'audio'
            else:
                return 'text'
        elif param.type == 'integer':
            return 'int'
        elif param.type == 'number':
            return 'float'
        elif param.type == 'boolean':
            return 'bool'
        elif param.type is None:
            # Handle unspecified type as simple string.
            return 'text'
        else:
            raise NotImplementedError(f'Unsupported type `{param.type}`.')

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self._parameters
