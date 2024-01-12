import typing
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from pydantic import BaseModel

if hasattr(typing, 'Annotated'):
    Annotated = typing.Annotated
else:
    from typing_extensions import Annotated
    Annotated = Annotated


class Parameter(BaseModel):
    """Meta information for parameters.

    Args:
        name (str): tool name for agent to identify the tool.
        category (str): Category of the parameter.
        description (Optional[str]): Description for the parameter.
            Defaults to None.
        optional (bool): Whether the parameter has a default value.
            Defaults to False.
        default (Any): The default value of the parameter.
    """
    type: type
    name: Optional[str] = None
    description: Optional[str] = None
    optional: bool = False
    default: Any = None


class ToolMeta(BaseModel):
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str): Description for tool.
        inputs (tuple[str, ...]): Input categories for tool.
        outputs (tuple[str, ...]): Output categories for tool.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    inputs: Optional[Tuple[Union[str, Parameter], ...]] = None
    outputs: Optional[Tuple[Union[str, Parameter], ...]] = None


@dataclass
class Info:
    """Used to add additional information of arguments and outputs."""
    description: Optional[str] = None
    name: Optional[str] = None
