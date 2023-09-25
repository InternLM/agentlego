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
from agentlego.types import AudioIO, ImageIO

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
                input_fields=tool.input_fields,
            ))
    return response


def add_tool(tool_name: str):
    tool: BaseTool = tools[tool_name]
    inputs = tool.toolmeta.inputs

    def _call(**kwargs):
        args = {}
        for name, in_category in zip(tool.input_fields, inputs):
            data = kwargs[name]
            if in_category == 'text':
                data = data
            elif in_category == 'image':
                from PIL import Image
                data = ImageIO(Image.open(data.file))
            elif in_category == 'audio':
                import torchaudio
                file_format = data.filename.rpartition('.')[-1] or None
                raw, sr = torchaudio.load(data.file, format=file_format)
                data = AudioIO(raw, sampling_rate=sr)
            args[name] = data

        outs = tool(**args)
        if not isinstance(outs, tuple):
            outs = [outs]

        res = []
        for out, out_category in zip(outs, tool.toolmeta.outputs):
            if out_category == 'text':
                res.append(out)
            elif out_category == 'image':
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
                raise NotImplementedError
        return res

    def call(**kwargs):
        try:
            return _call(**kwargs)
        except Exception as e:
            return dict(error=repr(e))

    call_args = {}
    call_params = []
    for arg_name, in_category in zip(tool.input_fields, inputs):
        if in_category == 'text':
            type_ = Annotated[str, Form()]
        elif in_category in ['image', 'audio']:
            type_ = Annotated[UploadFile, File(media_type=in_category)]
        call_args[arg_name] = type_
        call_params.append(
            inspect.Parameter(
                arg_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=type_))
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
            toolmeta=tool.toolmeta.__dict__, input_fields=tool.input_fields),
        methods=['GET'],
    )


for tool_name in tools:
    add_tool(tool_name)
app.include_router(tool_router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=args.port)
