# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Callable, Union

import mmlmtools.tools as tools
from mmlmtools.tools.base import BaseTool
from mmlmtools.utils.cache import load_or_build_object

NAMES2TOOLS = {
    k: v
    for k, v in tools.__dict__.items()
    if isinstance(v, type) and issubclass(v, BaseTool)
}


def import_all_tools_to(target_dir):
    global_dict = sys.modules[target_dir].__dict__
    for k, v in tools.__dict__.items():
        if isinstance(v, type) and issubclass(v, BaseTool):
            global_dict[k] = v


def list_tools():
    return NAMES2TOOLS.keys()


def load_tool(tool_name: str, **kwargs) -> Union[Callable, BaseTool]:
    """Load a configurable callable tool for different task.

    Args:
        tool_name (str): tool name for specific task. You can find more
            description about supported tools in `Capability Matrix`_
        **kwargs: key-word arguments to build the specific tools.
            These arguments are related ``tool``. You can find the arguments
            of the specific tool type according to the given tool in the
            `Capability Matrix`_

    Returns:
        callable: A callable tool.

    Examples:
        >>> from mmlmtools import load_tool
        >>> # load tool with tool name
        >>> tool, meta = load_tool('object detection')
        >>> # load a specific model
        >>> tool, meta = load_tool(
        >>>     'object detection', model='rtmdet_l_8xb32-300e_coco')

    Returns:
        Tuple[callable, ToolMeta]: _description_

    .. _Capability Matrix: TODO
    """
    if tool_name not in NAMES2TOOLS:
        # Using ValueError to show error msg cross lines.
        raise ValueError(f'{tool_name} is not supported now, the available '
                         'tools are:\n' +
                         '\n'.join(map(repr, NAMES2TOOLS.keys())))

    tool_type = NAMES2TOOLS[tool_name]
    tool_obj = load_or_build_object(tool_type, **kwargs)
    return tool_obj
