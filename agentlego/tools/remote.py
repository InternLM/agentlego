import base64
from io import BytesIO, IOBase
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

import requests

from agentlego.parsers import DefaultParser
from agentlego.schema import Parameter
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, File, ImageIO
from agentlego.utils.openapi import (APIOperation, APIResponseProperty, OpenAPISpec,
                                     operation_toolmeta)


class RemoteTool(BaseTool):
    """Create a tool from an OpenAPI Specification (OAS).

    It supports `OpenAPI v3.1.0 <https://spec.openapis.org/oas/latest.html#version-3-1-0>`_

    Examples:
        1. Construct a series of tools from an OAS.

        .. code::python
            from agentlego.tools import RemoteTool

            tools = RemoteTool.from_openapi('http://localhost:16180/openapi.json')

        In this situation, you need to provide the path or URL of an OAS, and each
        method will be constructed as a tool.

        2. Construct a single tool from URL.

        .. code::python
            from agentlego.tools import RemoteTool

            tool = RemoteTool.from_url('http://localhost:16180/ImageDescription')

        In this situation, you need to provide the URL of the tool endpoint.
        By default, it will get the OAS from ``http://localhost:16180/openapi.json``,
        and use the operation ``post`` at path ``/ImageDescription`` to construct the
        tool.

    Notice:
        The ``RemoteTool`` works well with the ``agentlego-server``.
    """  # noqa: E501

    def __init__(
        self,
        operation: APIOperation,
        headers: Optional[dict] = None,
        auth: Optional[tuple] = None,
        toolkit: Optional[str] = None,
    ):
        self.operation = operation
        self.url = urljoin(operation.base_url, operation.path)
        self.headers = headers
        self.auth = auth
        self.method = operation.method.name
        self.toolmeta = operation_toolmeta(operation)
        self.toolkit = toolkit
        self.set_parser(DefaultParser)
        self._is_setup = False

    def _construct_path(self, kwargs: Dict[str, str]) -> str:
        """Construct url according to path parameters from inputs."""
        path = self.url
        for param in self.operation.path_params:
            path = path.replace(f'{{{param}}}', str(kwargs.pop(param, '')))
        return path

    def _construct_query(self, kwargs: Dict[str, str]) -> Dict[str, str]:
        """Construct query parameters from inputs."""
        query_params = {}
        for param in self.operation.query_params:
            if param in kwargs:
                query_params[param] = kwargs.pop(param)
        return query_params

    def _construct_body(self, kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Construct request body parameters from inputs."""
        if not self.operation.request_body or not self.operation.body_params:
            return {}

        media_type = self.operation.request_body.media_type

        body = {}
        for param in self.operation.body_params:
            if param in kwargs:
                value = kwargs.pop(param)
                if isinstance(value, (ImageIO, AudioIO, File)):
                    value = value.to_file()
                body[param] = value

        if media_type == 'multipart/form-data':
            body = {
                k: (k, v) if isinstance(v, IOBase) else (None, v)
                for k, v in body.items()
            }
            return {'files': body}
        elif media_type == 'application/json':
            return {'json': body}
        elif media_type == 'application/x-www-form-urlencoded':
            return {'data': body}
        else:
            raise NotImplementedError(f'Unsupported media type `{media_type}`')

    @staticmethod
    def _parse_output(out: Any, p: Parameter):
        if p.type is ImageIO:
            file = BytesIO(base64.b64decode(out))
            out = ImageIO.from_file(file)
        elif p.type is AudioIO:
            file = BytesIO(base64.b64decode(out))
            out = AudioIO.from_file(file)
        elif p.type is File:
            file = BytesIO(base64.b64decode(out))
            out = File.from_file(file, filetype=p.filetype)
        return out

    def apply(self, *args, **kwargs):
        for arg, p in zip(args, self.inputs):
            kwargs[p.name] = arg

        request_args = {
            'url': self._construct_path(kwargs),
            'params': self._construct_query(kwargs),
            **self._construct_body(kwargs)
        }

        try:
            response = requests.request(
                method=self.method,
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
            raise RuntimeError(f'Failed to call the remote tool `{self.name}` '
                               f'because of {response.reason}.\nResponse: {content}')
        try:
            response = response.json()
        except requests.JSONDecodeError as e:
            raise RuntimeError(f'Failed to call the remote tool `{self.name}` '
                               'because of unknown response.\n'
                               f'Response: {response.content.decode()}') from e

        response_schema = self.operation.responses
        if response_schema is None or response_schema.get('200') is None:
            # Directly use string if the response schema is not specified
            return str(response)

        out_props = response_schema['200'].properties

        if isinstance(out_props, APIResponseProperty):
            # Single output
            return self._parse_output(response, self.outputs[0])
        elif isinstance(out_props, list):
            # Multiple output
            return tuple(
                self._parse_output(out, p) for out, p in zip(response, self.outputs))
        else:
            # Dict-style output
            return {
                p.name: self._parse_output(out, p)
                for out, p in zip(response, self.outputs)
            }

    @classmethod
    def from_server(cls, url: str, **kwargs) -> List['RemoteTool']:
        return cls.from_openapi(url=urljoin(url, '/openapi.json'), **kwargs)

    @classmethod
    def from_openapi(
        cls,
        url: str,
        **kwargs,
    ) -> List['RemoteTool']:
        """Construct a series of remote tools from the specified OpenAPI
        Specification.

        Args:
            url (str): The path or URL of the OpenAPI Specification file.
            headers (str | None): The headers to send in the requests. Defaults to None.
            auth (tuple | None): Auth tuple to enable Basic/Digest/Custom HTTP Auth.
                Defaults to None.
        """
        if url.startswith('http'):
            spec = OpenAPISpec.from_url(url)
        else:
            spec = OpenAPISpec.from_file(url)

        toolkit = spec.info.title.replace(' ', '_')

        tools = []
        for path, method in spec.iter_all_method():
            operation = APIOperation.from_openapi_spec(spec, path, method)
            tool = cls(
                operation=operation,
                toolkit=toolkit,
                **kwargs,
            )
            tools.append(tool)
        return tools

    @classmethod
    def from_url(cls,
                 url: str,
                 method: str = 'post',
                 openapi: Optional[str] = None,
                 path: Optional[str] = None,
                 **kwargs) -> 'RemoteTool':
        """Construct a remote tool from the specified URL endpoint.

        Args:
            url (str): The URL path of the remote tool.
            method (str): The method of the operation. Defaults to 'post',
            openapi (str | None): The OAS path or URL. Defaults to None, which means to
                use ``<base_url>/openapi.json``.
            path (str | None): The path in the OAS. Defaults to None, which means to use
                ``urlsplit(url).path``.
            headers (str | None): The headers to send in the requests. Defaults to None.
            auth (tuple | None): Auth tuple to enable Basic/Digest/Custom HTTP Auth.
                Defaults to None.
        """
        # The default openapi file for the tool server.
        openapi = openapi or urljoin(url, '/openapi.json')
        path = path or urlsplit(url).path

        if openapi.startswith('http'):
            spec = OpenAPISpec.from_url(openapi)
        else:
            spec = OpenAPISpec.from_file(openapi)

        toolkit = spec.info.title.replace(' ', '_')

        operation = APIOperation.from_openapi_spec(spec, path, method)
        return cls(operation=operation, toolkit=toolkit, **kwargs)
