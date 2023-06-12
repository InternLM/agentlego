# Copyright (c) OpenMMLab. All rights reserved.
import inspect

import mmlmtools.tools as tools
from mmlmtools.tools import *  # noqa: F401, F403
from mmlmtools.tools.base_tool import BaseTool


def load_tools_for_visualchatgpt_igpt(load_dict, models):
    for class_name, device in load_dict.items():
        if class_name in tools.__all__:
            v = globals()[class_name](device=device)
            v.inference.name = v.toolmeta.tool_name
            v.inference.description = v.toolmeta.description
            models[class_name] = v


def convert_tools_for_visualchatgpt_igpt(models):
    for v in models.values():
        if isinstance(v, BaseTool):
            v.inference.name = v.toolmeta.tool_name
            v.inference.description = v.toolmeta.description


def init_tools_for_visualchatgpt_igpt():
    for v in globals().values():
        if inspect.isclass(v) and issubclass(v, BaseTool):
            v.inference.name = v.DEFAULT_TOOLMETA.tool_name
            v.inference.description = v.DEFAULT_TOOLMETA.description


init_tools_for_visualchatgpt_igpt()
