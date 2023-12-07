from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str): Description for tool.
        inputs (tuple[str, ...]): Input categories for tool.
        outputs (tuple[str, ...]): Output categories for tool.
    """
    name: str
    description: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


@dataclass
class Parameter:
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
    name: str
    category: str
    description: Optional[str] = None
    optional: bool = False
    default: Any = None
