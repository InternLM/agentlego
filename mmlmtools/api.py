# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os.path as osp
import warnings
from collections import defaultdict
from importlib.util import find_spec
from pickle import dumps
from typing import Optional, Tuple, Union

import modelindex
from modelindex.models.Model import Model
from modelindex.models.ModelIndex import BaseModelIndex, ModelIndex
from rich.progress import track

from .utils import Mode, ToolMeta, _get_required_repos

DEFAULT_TOOLS = {}

# Loaded from OpenMMLab metafiles
MMTOOLS = defaultdict(dict)

# mapping between task name and Tool object.
TASK2TOOL = {}

CACHED_TOOLS = {}


def load_tool(tool: str,
              *,
              model: Optional[str] = None,
              mode: Union[str, Mode, None] = None,
              **kwargs) -> Tuple[callable, ToolMeta]:
    """Load a configurable callable tool for different task.

    Args:
        tool (str): tool name for specific task. You can find more description
            about supported tools in `Capability Matrix`_
        model (str, optional): model name defined in OpenMMLab metafile. If it
            is not specified, recommended tool will be loaded according to the
            ``tool``. You can find more description about supported model in
            `Capability Matrix`_. Defaults to None.
        mode (str, Mode, optional): If the model parameter is set to None,
            you have the option to select a recommended tool by configuring
            the argument with one of the following choices:

            - "efficiency": Load a high efficiency tool
            - "performance": Load a high performance tool
            - "balance": Load a tool that strikes a balance between the two

            mode could also be a ``Mode`` object. Defaults to None.
        **kwargs: key-word arguments to build the specific tools.
            These arguments are related ``tool``. You can find the arguments
            of the specific tool type according to the given tool in the
            `Capability Matrix`_

    Returns:
        Tuple[callable, ToolMeta]: A tuple with callable tool and its meta
        information. The commonly used information by LLM agent like
        "description" can be found in meta.

    Examples:
        >>> from mmlmtools import load_tool
        >>> # load tool with tool name
        >>> tool, meta = load_tool('object detection')
        >>> # load a high efficiency detection tool
        >>> tool, meta = load_tool('object detection', mode='efficiency')
        >>> # load a specific model
        >>> tool, meta = load_tool(
        >>>     'object detection', model='rtmdet_l_8xb32-300e_coco')



    Returns:
        Tuple[callable, ToolMeta]: _description_

    .. _Capability Matrix: TODO
    """
    if tool not in DEFAULT_TOOLS:
        # Using ValueError to show error msg cross lines.
        raise ValueError(f'{tool} is not supported now, the available '
                         'tools are:\n' + '\n'.join(map(repr, MMTOOLS.keys())))
    if isinstance(mode, Mode):
        mode = mode.name

    tool_meta: ToolMeta
    if model is None:
        tool_metas = DEFAULT_TOOLS[tool]
        if isinstance(tool_metas, dict):
            if mode is None:
                tool_meta = list(tool_metas.values())[0]
            else:
                if mode not in tool_metas:
                    raise ValueError(
                        f'{mode} is not available for {tool}, the available '
                        'modes are:\n' +
                        '\n'.join(map(repr, tool_metas.keys())))
                tool_meta = tool_metas[mode]
        else:
            if mode is not None:
                raise ValueError(
                    f'mode should not be configured for tool {tool}')
            else:
                tool_meta = tool_metas
    else:
        if mode is not None:
            raise ValueError(
                'mode should not be configured when model is specified')
        tool_metas = MMTOOLS[tool]
        if model not in tool_metas:
            raise ValueError(f'{model} is not a correct model name,'
                             f'the available model names for {tool} are\n' +
                             '\n'.join(map(repr, tool_metas.keys())))
        tool_meta = tool_metas[model]
    tool_type = tool_meta.tool_type
    tool_id = dumps((tool_meta.model, kwargs))
    if tool_id in CACHED_TOOLS:
        tool = CACHED_TOOLS[tool_id]
    else:
        if inspect.isclass(tool):
            tool = tool_type(tool_meta.model, **kwargs)
        else:
            # function tool
            tool = tool_type
    return tool, tool_meta


def register_custom_tool(*, tool, description, force=False):
    """Register custom tool.

    Examples:
        >>> @register_custom_tool(tool="python code executor, description="execute python code")
        >>> def python_code_executor(inputs:)
        >>>     ...
    """  # noqa: E501

    def wrapper(func):
        if tool not in DEFAULT_TOOLS:
            DEFAULT_TOOLS[tool] = ToolMeta(
                tool_type=func, description=description)
        else:
            if not force:
                raise KeyError(
                    'Please do not register tool with duplicated name '
                    f'{tool}. If you want to overwrite the old tool, please '
                    'set `force=True`')
            else:
                DEFAULT_TOOLS[tool] = ToolMeta(
                    tool_type=func, description=description)
        return func

    return wrapper


def collect_tools():
    """Initialize MMTOOLS."""
    global MMTOOLS
    MMTOOLS.clear()
    repos = _get_required_repos()
    """Collect tools from metafile"""

    def _model_index_to_dict(item):
        if isinstance(item, BaseModelIndex):
            if isinstance(item.data, dict):
                return {
                    key: (_model_index_to_dict(value))
                    for key, value in item.data.items()
                }
            elif isinstance(item.data, (list, tuple)):
                return type(item.data)(_model_index_to_dict(i) for i in item)
            elif isinstance(item.data, BaseModelIndex):
                return _model_index_to_dict(item.data)
        else:
            return item

    for repo in repos:
        spec = find_spec(repo)
        if spec is None:
            warnings.warn(f'Local tools from {repo} will not be loaded')
            continue
        package_path = spec.submodule_search_locations[0]

        # Load model-index from source code
        if osp.exists(osp.join(osp.dirname(package_path), 'model-index.yml')):
            model_index_path = osp.join(
                osp.dirname(package_path), 'model-index.yml')
        # Load model-index from .mim in installed package.
        else:
            mim_path = osp.join(package_path, '.mim')
            model_index_path = osp.join(mim_path, 'model-index.yml')

        model_index: ModelIndex = modelindex.load(model_index_path)
        model_index.build_models_with_collections()

        model: Model
        for model in track(
                model_index.models, description=f'load tools from {repo}'):
            if model.results is None:
                continue
            model = _model_index_to_dict(model.full_model)
            collection_name = model['In Collection']
            model_name = model['Name'].lower()
            for result in model['Results']:
                task = result['Task'].lower()
                if task not in TASK2TOOL:
                    continue
                description = model.get('Description',
                                        f'{task} tool: {collection_name}')
                MMTOOLS[task][model_name] = ToolMeta(
                    tool_type=TASK2TOOL[task],
                    model=model_name,
                    description=description)
