import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type


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
    type: Optional[Type] = None
    name: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None
    default: Optional[Any] = None
    filetype: Optional[str] = None

    def update(self, other: 'Parameter'):
        other = copy.deepcopy(other)
        for k, v in copy.deepcopy(other.__dict__).items():
            if v is not None:
                self.__dict__[k] = v


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
    inputs: Optional[Tuple[Parameter, ...]] = None
    outputs: Optional[Tuple[Parameter, ...]] = None
