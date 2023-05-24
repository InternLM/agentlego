# Copyright (c) OpenMMLab. All rights reserved.
from .api import collect_tools, load_tool, register_custom_tool

collect_tools()

__all__ = ['load_tool', 'collect_tools', 'register_custom_tool']
