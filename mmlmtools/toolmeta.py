# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        tool_name (callable): name of tool (task). If there are more than one
            tools built for the same task with different arguments, tool_name
            will end up with a index suffix like "object detection {index}"
        description (str, optional): Description for tool. Defaults to None
    """
    tool_name: str
    description: Optional[str] = None
    model: Optional[str] = None
    input_description: Optional[str] = None
    output_description: Optional[str] = None
