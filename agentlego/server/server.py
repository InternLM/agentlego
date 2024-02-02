import base64
import inspect
import logging
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

from agentlego.apis.tool import (extract_all_tools, list_tools, load_tool,
                                 register_all_tools)
from agentlego.parsers import NaiveParser
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO
from agentlego.types import File as FileType
from agentlego.types import ImageIO
from agentlego.utils import resolve_module

try:
    import rich
    import typer
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import RedirectResponse
    from makefun import create_function
    from pydantic import Field
    from rich.table import Table
    from typing_extensions import Annotated
except ImportError:
    print('[Import Error] Failed to import server dependencies, '
          'please install by `pip install agentlego[server]`')
    sys.exit(1)

cli = typer.Typer(add_completion=False, no_args_is_help=True)


def create_input_params(tool: BaseTool) -> List[inspect.Parameter]:
    params = []
    for p in tool.inputs:
        field_kwargs = {}
        if p.description:
            field_kwargs['description'] = p.description
        if p.type is ImageIO:
            field_kwargs['format'] = 'image;binary'
            annotation = Annotated[UploadFile, File(**field_kwargs)]
        elif p.type is AudioIO:
            field_kwargs['format'] = 'audio;binary'
            annotation = Annotated[UploadFile, File(**field_kwargs)]
        elif p.type is FileType:
            filetype = p.filetype or 'file'
            field_kwargs['format'] = f'{filetype};binary'
            annotation = Annotated[UploadFile, File(**field_kwargs)]
        else:
            annotation = Annotated[p.type, Form(**field_kwargs)]

        param = inspect.Parameter(
            p.name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=p.default if p.optional else inspect._empty,
            annotation=annotation,
        )
        params.append(param)

    return params


def create_output_annotation(tool: BaseTool):
    output_schema = []

    for p in tool.outputs:
        field_kwargs = {}
        if p.description:
            field_kwargs['description'] = p.description
        if p.type is ImageIO:
            annotation = str
            field_kwargs['format'] = 'image/png;base64'
        elif p.type is AudioIO:
            annotation = str
            field_kwargs['format'] = 'audio/wav;base64'
        elif p.type is FileType:
            annotation = str
            filetype = p.filetype or 'file'
            field_kwargs['format'] = f'{filetype};base64'
        else:
            assert p.type is not None
            annotation = p.type

        output_schema.append(Annotated[annotation, Field(**field_kwargs)])

    if len(output_schema) == 0:
        return None
    elif len(output_schema) == 1:
        return output_schema[0]
    else:
        return Tuple.copy_with(tuple(output_schema))


def add_tool(tool: BaseTool, app: FastAPI):
    tool_name = tool.name.replace(' ', '_')

    input_params = create_input_params(tool)
    return_annotation = create_output_annotation(tool)
    signature = inspect.Signature(input_params, return_annotation=return_annotation)

    def _call(**kwargs):
        args = {}
        for p in tool.inputs:
            data = kwargs[p.name]
            if p.type is ImageIO:
                from PIL import Image
                data = ImageIO(Image.open(data.file))
            elif p.type is AudioIO:
                import torchaudio
                file_format = data.filename.rpartition('.')[-1] or None
                raw, sr = torchaudio.load(data.file, format=file_format)
                data = AudioIO(raw, sampling_rate=sr)
            elif p.type is FileType:
                data = FileType(data.file.read())
            elif data is None:
                continue
            else:
                data = p.type(data)
            args[p.name] = data

        outs = tool(**args)
        if not isinstance(outs, tuple):
            outs = [outs]

        res = []
        for out, p in zip(outs, tool.outputs):
            if p.type is ImageIO:
                file = BytesIO()
                out.to_pil().save(file, format='png')
                out = base64.b64encode(file.getvalue()).decode()
            elif p.type is AudioIO:
                import torchaudio
                file = BytesIO()
                torchaudio.save(file, out.to_tensor(), out.sampling_rate, format='wav')
                out = base64.b64encode(file.getvalue()).decode()
            elif p.type is FileType:
                out = base64.b64encode(out.to_bytes()).decode()
            res.append(out)

        if len(res) == 0:
            return None
        elif len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    def call(**kwargs):
        try:
            return _call(**kwargs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=repr(e))

    app.add_api_route(
        f'/{tool_name}',
        endpoint=create_function(signature, call),
        methods=['POST'],
        operation_id=tool_name,
        summary=tool.toolmeta.description,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger('uvicorn.error')
    logger.info(f'OpenAPI spec file at \x1b[1m{app.openapi_url}\x1b[0m')
    yield


@cli.command(no_args_is_help=True)
def start(
        tools: List[str] = typer.Argument(
            help='The class name of tools to deploy.', show_default=False),
        device: str = typer.Option(
            'cuda:0', help='The device to use to deploy the tools.'),
        setup: bool = typer.Option(True, help='Setup tools during starting the server.'),
        extra: Optional[List[Path]] = typer.Option(
            None,
            help='The extra Python source files or modules includes tools.',
            file_okay=True,
            dir_okay=True,
            exists=True,
            show_default=False),
        host: str = typer.Option('127.0.0.1', help='The server address.'),
        port: int = typer.Option(16180, help='The server port.'),
        title: str = typer.Option('AgentLego', help='The title of the tool collection.'),
):
    """Start a tool server with the specified tools."""
    app = FastAPI(title=title, openapi_url='/openapi.json', lifespan=lifespan)

    @app.get('/', include_in_schema=False)
    async def root():
        return RedirectResponse(url='/openapi.json')

    if extra is not None:
        for path in extra:
            register_all_tools(resolve_module(path))

    for name in tools:
        tool = load_tool(name, device=device)
        tool.set_parser(NaiveParser)
        if setup:
            tool.setup()
            tool._is_setup = True

        add_tool(tool, app)

    uvicorn.run(app, host=host, port=port)


@cli.command(name='list')
def list_available_tools(
    official: bool = typer.Option(
        True, help='Whether to show AgentLego official tools.'),
    extra: Optional[List[Path]] = typer.Option(
        None,
        help='The extra Python source files or modules includes tools.',
        exists=True,
        show_default=False,
        resolve_path=True),
):
    """List all available tools."""

    table = Table('Class', 'source')
    if official:
        for name in sorted(list_tools()):
            table.add_row(name, '[green]Official[/green]')

    if extra is not None:
        for path in extra:
            names2tools = extract_all_tools(resolve_module(path))
            for name in sorted(list(names2tools.keys())):
                table.add_row(name, str(path))
    rich.print(table)


if __name__ == '__main__':
    cli()
