import argparse
import base64
import inspect
from io import BytesIO
from typing import Dict
from urllib.parse import quote_plus

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from typing_extensions import Annotated

from agentlego.apis import load_tool
from agentlego.parsers import NaiveParser
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, CatgoryToIO, ImageIO

prog_description = """\
Start a server for several tools.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'tools',
        type=str,
        nargs='+',
        help='The tools to deploy',
    )
    parser.add_argument(
        '--port',
        default=16180,
        type=int,
        help='The port number',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help='The device to deploy the tools',
    )
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Avoid setup tools during starting the server.',
    )
    args = parser.parse_args()
    return args


args = parse_args()
tools: Dict[str, BaseTool] = {}
for name in args.tools:
    tool = load_tool(name, device=args.device, parser=NaiveParser)
    if not args.no_setup:
        tool.setup()
        tool._is_setup = True
    tools[quote_plus(tool.name.replace(' ', ''))] = tool

app = FastAPI()
tool_router = APIRouter()


@app.get('/')
def index():
    response = []
    for tool_name, tool in tools.items():
        response.append(
            dict(
                domain=tool_name,
                toolmeta=tool.toolmeta.__dict__,
                parameters=[p.__dict__ for p in tool.parameters.values()],
            ))
    return response


def add_tool(tool_name: str):
    tool: BaseTool = tools[tool_name]

    def _call(**kwargs):
        args = {}
        for p in tool.parameters.values():
            data = kwargs[p.name]
            if p.category == 'image':
                from PIL import Image
                data = ImageIO(Image.open(data.file))
            elif p.category == 'audio':
                import torchaudio
                file_format = data.filename.rpartition('.')[-1] or None
                raw, sr = torchaudio.load(data.file, format=file_format)
                data = AudioIO(raw, sampling_rate=sr)
            else:
                data = CatgoryToIO[p.category](data)
            args[p.name] = data

        outs = tool(**args)
        if not isinstance(outs, tuple):
            outs = [outs]

        res = []
        for out, out_category in zip(outs, tool.toolmeta.outputs):
            if out_category == 'image':
                file = BytesIO()
                out.to_pil().save(file, format='png')
                res.append(
                    dict(
                        type='image',
                        data=base64.encodebytes(
                            file.getvalue()).decode('ascii'),
                    ))
            elif out_category == 'audio':
                import torchaudio
                file = BytesIO()
                torchaudio.save(
                    file, out.to_tensor(), out.sampling_rate, format='wav')
                res.append(
                    dict(
                        type='audio',
                        data=base64.encodebytes(
                            file.getvalue()).decode('ascii'),
                    ))
            else:
                res.append(out)
        return res

    def call(**kwargs):
        try:
            return _call(**kwargs)
        except Exception as e:
            return dict(error=repr(e))

    call_args = {}
    call_params = []
    for p in tool.parameters.values():
        if p.category in ['image', 'audio']:
            annotation = Annotated[UploadFile, File(media_type=p.category)]
        else:
            type_ = {
                'text': str,
                'int': int,
                'bool': bool,
                'float': float
            }[p.category]
            annotation = Annotated[type_, Form()]

        call_args[p.name] = annotation
        call_params.append(
            inspect.Parameter(
                p.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=p.default if p.optional else inspect._empty,
                annotation=annotation,
            ))
    call.__signature__ = inspect.Signature(call_params)
    call.__annotations__ = call_args
    tool_router.add_api_route(
        f'/{tool_name}/call',
        endpoint=call,
        methods=['POST'],
    )
    tool_router.add_api_route(
        f'/{tool_name}/meta',
        endpoint=lambda: dict(
            toolmeta=tool.toolmeta.__dict__,
            parameters=[p.__dict__ for p in tool.parameters.values()]),
        methods=['GET'],
    )


for tool_name in tools:
    add_tool(tool_name)
app.include_router(tool_router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=args.port)
