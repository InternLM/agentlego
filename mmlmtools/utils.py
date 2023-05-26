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
        tool_type (callable): class object to build the tool or callable tool.
        model (str, optional): model name for OpenMMLab tools.
            Defaults to None.
        description (str, optional): Description for tool. Defaults to None
    """
    tool_type: callable
    model: Optional[str] = None
    description: Optional[str] = None
