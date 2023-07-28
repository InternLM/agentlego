# Copyright (c) OpenMMLab. All rights reserved.

from mmlmtools.api import import_all_tools_to
from mmlmtools.tools import *  # noqa: F401, F403
from .visual_chatgpt import (convert_mmtools_for_visualchatgpt,
                             load_mmtools_for_visualchatgpt)


def load_mmtools_for_igpt(load_dict):
    return load_mmtools_for_visualchatgpt(load_dict)


def convert_mmtools_for_igpt(models):
    convert_mmtools_for_visualchatgpt(models)


# global_dict = sys.modules['iGPT.controllers.ConversationBot'].__dict__
# for k, v in tools.__dict__.items():
#     if inspect.isclass(v) and issubclass(v, BaseToolv1):
#         global_dict[k] = v
import_all_tools_to('iGPT.controllers.ConversationBot')
