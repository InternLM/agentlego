# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str, optional): Description for tool. Defaults to None
        model (str, optional): Model name for tool. Defaults to None
        input_description (str, optional): Input description for tool.
            Defaults to None
        output_description (str, optional): Output description for tool.
            Defaults to None
    """
    name: str
    description: Optional[str] = None
    model: Optional[str] = None
    input_description: Optional[str] = None
    output_description: Optional[str] = None
