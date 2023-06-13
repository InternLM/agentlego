# Copyright (c) OpenMMLab. All rights reserved.
import mmlmtools.tools as tools
from mmlmtools.api import import_all_tools_to
from mmlmtools.tools import *  # noqa: F401, F403
from mmlmtools.tools.base_tool import BaseTool


def load_tools_for_visualchatgpt(load_dict):
    models = {}
    for class_name, device in load_dict.items():
        if class_name in tools.__all__:
            v = globals()[class_name](device=device)
            v.inference.__dict__['name'] = v.toolmeta.name
            v.inference.__dict__['description'] = v.toolmeta.description
            models[class_name] = v
    return models


def convert_tools_for_visualchatgpt(models):
    for k, v in models.items():
        if isinstance(v, BaseTool):
            v.inference.__dict__['name'] = v.toolmeta.name
            v.inference.__dict__['description'] = v.toolmeta.description
            models[k] = v


# global_dict = sys.modules['visual_chatgpt.ConversationBot'].__dict__
# for k, v in tools.__dict__.items():
#     if inspect.isclass(v) and issubclass(v, BaseTool):
#         global_dict[k] = v
import_all_tools_to('__main__')
