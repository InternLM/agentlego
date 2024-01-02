import argparse
import base64
import inspect
from io import BytesIO
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from makefun import create_function
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined
from typing_extensions import Annotated

from agentlego.apis import load_tool
from agentlego.parsers import NaiveParser
from agentlego.tools.base import BaseTool
from agentlego.types import AudioIO, CatgoryToIO, ImageIO

prog_description = """\
Start a server for the specified tools.
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


def create_input_params(tool: BaseTool) -> List[inspect.Parameter]:
    params = []
    for p in tool.parameters.values():
        field_kwargs = {}
        if p.description:
            field_kwargs['description'] = p.description
        if p.category in ['image', 'audio']:
            field_kwargs['format'] = p.category + ';binary'
            annotation = Annotated[UploadFile, File(**field_kwargs)]
        else:
            annotation = Annotated[CatgoryToIO[p.category],
                                   Form(**field_kwargs)]

        param = inspect.Parameter(
            p.name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=p.default if p.optional else inspect._empty,
            annotation=annotation,
        )
        params.append(param)

    return params


def create_output_model(tool: BaseTool) -> BaseModel:
    output_schema = []

    for category in tool.toolmeta.outputs:
        field_kwargs = {}
        if category == 'image':
            annotation = str
            field_kwargs['format'] = 'image/png;base64'
        elif category == 'audio':
            annotation = str
            field_kwargs['format'] = 'audio/wav;base64'
        else:
            annotation = CatgoryToIO[category]

        output_schema.append(Annotated[annotation, Field(**field_kwargs)])

    if len(output_schema) == 0:
        return None
    elif len(output_schema) == 1:
        return output_schema[0]
    else:
        return Tuple[*output_schema]


def add_tool(tool: BaseTool, router: APIRouter):
    tool_name = tool.name.replace(' ', '_')

    input_params = create_input_params(tool)
    output_model = create_output_model(tool)
    signature = inspect.Signature(input_params, return_annotation=output_model)

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
        for out, category in zip(outs, tool.toolmeta.outputs):
            if category == 'image':
                file = BytesIO()
                out.to_pil().save(file, format='png')
                out = base64.b64encode(file.getvalue()).decode()
            elif category == 'audio':
                import torchaudio
                file = BytesIO()
                torchaudio.save(
                    file, out.to_tensor(), out.sampling_rate, format='wav')
                out = base64.b64encode(file.getvalue()).decode()
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
            return dict(error=repr(e))

    router.add_api_route(
        f'/{tool_name}',
        endpoint=create_function(signature, call),
        methods=['POST'],
        operation_id=tool_name,
        description=tool.toolmeta.description,
    )


if __name__ == '__main__':
    args = parse_args()
    app = FastAPI()

    for name in args.tools:
        tool = load_tool(name, device=args.device, parser=NaiveParser)
        if not args.no_setup:
            tool.setup()
            tool._is_setup = True

        add_tool(tool, app)

    uvicorn.run(app, host='0.0.0.0', port=args.port)
