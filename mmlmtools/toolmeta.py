# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from typing import Optional


class Mode(enum.Enum):
    efficiency = 'high efficiency'
    balance = 'balance'
    performance = 'high performance'


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
