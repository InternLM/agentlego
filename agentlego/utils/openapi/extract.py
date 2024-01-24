import warnings
from typing import Tuple

from agentlego.schema import Parameter, ToolMeta
from .api_model import PRIMITIVE_TYPES, APIOperation, APIPropertyBase


def prop_to_parameter(prop: APIPropertyBase) -> Parameter:
    from agentlego.types import AudioIO, File, ImageIO
    p_type = PRIMITIVE_TYPES.get(prop.type, prop.type)  # type: ignore
    p = Parameter(
        type=p_type,
        name=prop.name if prop.name != '_null' else None,
        description=prop.description,
        optional=not prop.required,
        default=prop.default,
    )
    if p_type is str:
        schema_format = prop.format or ''
        if 'image' in schema_format:
            p.type = ImageIO
        elif 'audio' in schema_format:
            p.type = AudioIO
        elif 'binary' in schema_format or 'base64' in schema_format:
            p.type = File
            p.filetype, _, _ = schema_format.partition(';')
    return p


def operation_inputs(op: APIOperation) -> Tuple[Parameter, ...]:
    inputs = []
    properties = []
    if op.properties:
        properties.extend(op.properties)
    if op.request_body and op.request_body.properties:
        properties.extend(op.request_body.properties)
    for p in properties:
        inputs.append(prop_to_parameter(p))
    return tuple(inputs)


def operation_outputs(op: APIOperation) -> Tuple[Parameter, ...]:
    if op.responses is None or op.responses.get('200') is None:
        # If not specify outputs, directly handle as a single text.
        outputs = [Parameter(type=str)]

    response_schema = op.responses
    if response_schema is None or response_schema.get('200') is None:
        # Directly use string if the response schema is not specified
        warnings.warn(f'The response of {op.operation_id} is not specified, '
                      'assume as a string response by default.')
        return (Parameter(type=str), )
    else:
        out_props = response_schema['200'].properties

    if isinstance(out_props, list):
        outputs = [prop_to_parameter(out) for out in out_props]
    elif isinstance(out_props, dict):
        outputs = [prop_to_parameter(out) for out in out_props.values()]
    else:
        outputs = [prop_to_parameter(out_props)]

    return tuple(outputs)


def operation_toolmeta(operation: APIOperation) -> ToolMeta:
    """Extract tool meta information from a HTTP operation."""
    name = operation.operation_id
    inputs = operation_inputs(operation)
    outputs = operation_outputs(operation)
    toolmeta = ToolMeta(
        name=name,
        description=operation.description,
        inputs=inputs,
        outputs=outputs,
    )

    return toolmeta
