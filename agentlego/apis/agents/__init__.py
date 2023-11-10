# Copyright (c) OpenMMLab. All rights reserved.
from agentlego.utils import is_package_available
from .visual_chatgpt import load_tools_for_visual_chatgpt

__all__ = ['load_tools_for_visual_chatgpt']

if is_package_available('transformers'):
    from .huggingface_agent import load_tools_for_hfagent  # noqa: F401
    __all__.append('load_tools_for_hfagent')

if is_package_available('langchain'):
    from .langchain import load_tools_for_langchain  # noqa: F401
    __all__.append('load_tools_for_langchain')

if is_package_available('lagent'):
    from .lagent import load_tools_for_lagent  # noqa: F401
    __all__.append('load_tools_for_lagent')
