# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from mmlmtools.api import import_all_tools_to, list_tools
from mmlmtools.tools.base_tool import BaseTool


class Adapter:
    """Adapter for mmtools to Visual ChatGPT style.

    Args:
        tool (BaseTool): mmtool
    """

    def __init__(self, tool):
        self.tool = tool

    def __call__(self, inputs):
        return self.tool(inputs)

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
    mmtool_list = list_tools()
    for class_name, device in load_dict.items():
        if class_name in mmtool_list:
            v = globals()[class_name](device=device)
            v.inference = Adapter(v)
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
            v.inference = Adapter(v)
            v.inference.name = v.toolmeta.name
            v.inference.description = v.toolmeta.description
            models[k] = v


# global_dict = sys.modules['__main__'].__dict__
# for k, v in tools.__dict__.items():
#     if inspect.isclass(v) and issubclass(v, BaseTool):
#         global_dict[k] = v
import_all_tools_to('__main__')
