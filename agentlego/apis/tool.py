# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect

import agentlego.tools
from agentlego.tools import BaseTool
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


def list_tools(with_description=False):
    """List all the registered tools.

    Args:
        with_description (bool): Whether to return the description of tools.
            Defaults to False.

    Returns:
        list: list of tool names by default, or list of tuples
        `(tool_name, description)` if ``with_description=True``.

    Examples:
        >>> from agentlego import list_tools
        >>> # list all tools with description
        >>> for name, description in list_tools(with_description=True):
        ...     print(name, description)
    """
    if with_description:
        return list((name, cls.DEFAULT_TOOLMETA.description)
                    for name, cls in NAMES2TOOLS.items())
    else:
        return list(NAMES2TOOLS.keys())


def load_tool(tool_type: str,
              name: str = None,
              description: str = None,
              device=None,
              **kwargs) -> BaseTool:
    """Load a configurable callable tool for different task.

    Args:
        tool_name (str): tool name for specific task. You can find more
            description about supported tools by
            :func:`~agentlego.apis.list_tools`.
        override_name (str | None): The name to override the default name.
            Defaults to None.
        description (str): The description to override the default description.
            Defaults to None.
        device (str): The device to load the tool. Defaults to None.
        **kwargs: key-word arguments to build the specific tools.
            These arguments are related ``tool``. You can find the arguments
            of the specific tool type according to the given tool in the
            documentations of the tools.

    Returns:
        BaseTool: The constructed tool.

    Examples:
        >>> from agentlego import load_tool
        >>> # load tool with tool name
        >>> tool, meta = load_tool('object detection')
        >>> # load a specific model
        >>> tool, meta = load_tool(
        >>>     'object detection', model='rtmdet_l_8xb32-300e_coco')
    """
    if tool_type not in NAMES2TOOLS:
        # Using ValueError to show error msg cross lines.
        raise ValueError(f'{tool_type} is not supported now, the available '
                         'tools are:\n' + '\n'.join(NAMES2TOOLS.keys()))

    tool_type = NAMES2TOOLS[tool_type]
    if 'device' in inspect.getfullargspec(tool_type).args:
        kwargs['device'] = device

    if name or description:
        tool_obj = tool_type(**kwargs)
        if name:
            tool_obj.name = name
        if description:
            tool_obj.description = description
    else:
        # Only enable cache if no overrode attribution
        # to avoid the cached tool is changed.
        tool_obj = load_or_build_object(tool_type, **kwargs)
    return tool_obj
