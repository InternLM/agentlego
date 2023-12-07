from typing import Callable

CACHED_OBJECTS = {}


def load_or_build_object(constructor: Callable, *args, **kwargs):
    tool_id = str((constructor.__qualname__, args, kwargs))
    if tool_id in CACHED_OBJECTS:
        return CACHED_OBJECTS[tool_id]
    else:
        tool = constructor(*args, **kwargs)
        CACHED_OBJECTS[tool_id] = tool
        return tool
