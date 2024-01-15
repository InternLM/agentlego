import copy
import typing
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

if hasattr(typing, 'Annotated'):
    Annotated = typing.Annotated
else:
    from typing_extensions import Annotated
    Annotated = Annotated


@dataclass
class Parameter:
    """Meta information for parameters.

    Args:
        type (type): The type of the value.
        name (str): tool name for agent to identify the tool.
        description (str): Description for the parameter.
        optional (bool): Whether the parameter has a default value.
            Defaults to False.
        default (Any): The default value of the parameter.
    """
    type: Optional[type] = None
    name: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None
    default: Optional[Any] = None

    def update(self, other: 'Parameter'):
        other = copy.deepcopy(other)
        if other.type is not None:
            self.type = other.type
        if other.name is not None:
            self.name = other.name
        if other.description is not None:
            self.description = other.description
        if other.optional is not None:
            self.optional = other.optional
        if other.default is not None:
            self.default = other.default


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str): Description for tool.
        inputs (tuple[str | Parameter, ...]): Input categories for tool.
        outputs (tuple[str | Parameter, ...]): Output categories for tool.
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
