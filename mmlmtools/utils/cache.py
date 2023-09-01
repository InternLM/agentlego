# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

CACHED_OBJECTS = {}


def load_or_build_object(cls: Callable, *args, **kwargs):
    tool_id = str((cls.__name__, args, kwargs))
    if tool_id in CACHED_OBJECTS:
        return CACHED_OBJECTS[tool_id]
    else:
        tool = cls(*args, **kwargs)
        CACHED_OBJECTS[tool_id] = tool
        return tool
