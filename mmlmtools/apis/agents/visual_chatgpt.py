# Copyright (c) OpenMMLab. All rights reserved.

import inspect
import weakref

from mmlmtools import tools
from mmlmtools.tools.base_tool import BaseTool
from mmlmtools.tools.parsers import VisualChatGPTParser


def wrapped_init(self, *args, parser=VisualChatGPTParser(), **kwargs):
    return self.__class__.__bases__[0].__init__(
        self, *args, parser=VisualChatGPTParser(), **kwargs)


class _Inference:

    def __get__(self, instance, owner):
        if not hasattr(self, 'instance'):
            self.tool = weakref.ref(instance)
        return self

    @property
    def name(self):
        return self.tool().name

    @property
    def description(self):
        return self.tool().description

    def __call__(self, *args, **kwargs):
        return self.tool()(*args, **kwargs)


def load_tools_for_visual_chatgpt():
    """Load a set of tools and adapt them to Visual ChatGPT style.

    Args:
        tool_names (list[str]): list of tool names
        device (str): device to load tools. Defaults to 'cpu'.

    Returns:
    list(Tool): loaded tools
    """
    all_tools = inspect.getmembers(
        tools, lambda x: inspect.isclass(x) and issubclass(x, BaseTool))
    return {
        name: type(tool_cls.__name__, (tool_cls, ), {
            'inference': _Inference(),
            '__init__': wrapped_init
        })
        for name, tool_cls in all_tools
    }
