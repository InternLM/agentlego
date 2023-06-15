# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import mmlmtools.tools as tools
from mmlmtools.api import import_all_tools_to
from mmlmtools.tools import *  # noqa: F401, F403
from mmlmtools.tools.base_tool import BaseTool


class Adapter:
    """Adapter for mmtools to Visual ChatGPT style.

    Args:
        tool (BaseTool): mmtool
    """

    def __init__(self, tool):
        self.tool = tool

    # def __call__(self, inputs):
    #     return self.tool(inputs)

    def __get__(self, instance, owner):
        if not hasattr(self, 'adapter'):
            self.adapter = partial(self.tool, instance)
        return self.adapter


def load_mmtools_for_visualchatgpt(load_dict):
    """Load mmtools into Visual ChatGPT style.

    Args:
        load_dict (dict): dict of mmtools

    Returns:
        dict: dict of mmtools
    """
    models = {}
    for class_name, device in load_dict.items():
        if class_name in tools.__all__:
            v = globals()[class_name](device=device)
            v.inference = Adapter(v.__call__)
            # v.inference.__dict__['name'] = v.toolmeta.name
            # v.inference.__dict__['description'] = v.toolmeta.description
            v.inference.name = v.toolmeta.name
            v.inference.description = v.toolmeta.description
            models[class_name] = v
    return models


def convert_mmtools_for_visualchatgpt(models):
    """Convert mmtools into Visual ChatGPT style.

    Args:
        models (dict): dict of mmtools
    """
    for k, v in models.items():
        if isinstance(v, BaseTool):
            v.inference = Adapter(v.__call__)
            # v.inference.__dict__['name'] = v.toolmeta.name
            # v.inference.__dict__['description'] = v.toolmeta.description
            v.inference.name = v.toolmeta.name
            v.inference.description = v.toolmeta.description
            models[k] = v


# global_dict = sys.modules['__main__'].__dict__
# for k, v in tools.__dict__.items():
#     if inspect.isclass(v) and issubclass(v, BaseTool):
#         global_dict[k] = v
import_all_tools_to('__main__')
