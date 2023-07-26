# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str): Description for tool.
        model (dict, optional): Model dict for tool. Defaults to None
        input_description (str, optional): Input description for tool.
            Defaults to None
        output_description (str, optional): Output description for tool.
            Defaults to None
        inputs (tuple[str, ...], optional): Input categories for tool.
            Defaults to None
        outputs (tuple[str, ...], optional): Output categories for tool.
            Defaults to None
    """
    name: str
    description: str
    model: Optional[dict] = None
    input_description: Optional[str] = None
    output_description: Optional[str] = None
    inputs: Optional[tuple[str, ...]] = None
    outputs: Optional[tuple[str, ...]] = None
