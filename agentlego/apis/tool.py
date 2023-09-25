# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect

import agentlego.tools
from agentlego.tools.base import BaseTool
from agentlego.utils.cache import load_or_build_object

NAMES2TOOLS = {}


def register_all_tools(module):
    if isinstance(module, str):
        module = importlib.import_module(module)

    for k, v in module.__dict__.items():
        if (isinstance(v, type) and issubclass(v, BaseTool)
                and (v is not BaseTool)):
            NAMES2TOOLS[k] = v


register_all_tools(agentlego.tools)


def list_tools():
    return NAMES2TOOLS.keys()


def load_tool(tool_name: str, device=None, **kwargs) -> BaseTool:
    """Load a configurable callable tool for different task.

    Args:
        tool_name (str): tool name for specific task. You can find more
            description about supported tools in `Capability Matrix`_
        device (str): The device to load the tool. Defaults to None.
        **kwargs: key-word arguments to build the specific tools.
            These arguments are related ``tool``. You can find the arguments
            of the specific tool type according to the given tool in the
            `Capability Matrix`_

    Returns:
        callable: A callable tool.

    Examples:
        >>> from agentlego import load_tool
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
                         'tools are:\n' + '\n'.join(NAMES2TOOLS.keys()))

    tool_type = NAMES2TOOLS[tool_name]
    if 'device' in inspect.getfullargspec(tool_type).args:
        kwargs['device'] = device
    tool_obj = load_or_build_object(tool_type, **kwargs)
    return tool_obj
