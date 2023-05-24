# Copyright (c) OpenMMLab. All rights reserved.
import enum
import os.path as osp
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from importlib.util import find_spec
from typing import Optional

import modelindex
from modelindex.models.Model import Metadata, Model
from modelindex.models.ModelIndex import BaseModelIndex, ModelIndex

from .lmtools.base_tool import BaseTool

REPOS = [
    'mmagic',
    'mmpretrain',
    'mmocr',
    'mmdet'
]


DEFAULT_TOOLS = {}


TOOLS = defaultdict(dict)

task2tool = {

}




class Mode(enum.Enum):
    efficiency = 'high efficiency'
    balance = 'balance'
    performance = 'high performance'


@dataclass
class ToolMeta:
    tool_type: BaseTool
    model: Optional[str] = None
    description: Optional[str] = None
    mode: Optional[Mode] = None


def _collect_tools():
    def _model_index_to_dict(item):
        if isinstance(item, BaseModelIndex):
            if isinstance(item.data, dict):
                return {
                    key: (_model_index_to_dict(value))
                    for key, value in item.data.items()}
            elif isinstance(item.data, (list, tuple)):
                return type(item.data)(_model_index_to_dict(i) for i in item)
            elif isinstance(item.data, BaseModelIndex):
                return _model_index_to_dict(item.data)
        else:
            return item

    def _build_meta_from_formatted_info(info, formatted_info, task):
        """Formatted description with `Mode` will have an extra field in
        the TOOLS which means its collection. For example,
        `rtmdet_tiny_8xb32-300e_coco` is the most efficient model in RTMDet
        collection. Its metafile can be accessed in two ways:

            - TOOLS['object detection']['rtmdet_tiny_8xb32-300e_coco']
            - TOOLS['rtmdet']['efficiency']
        """
        global TOOLS
        for origin, formatted in zip(info, formatted_info):
            new_desc = formatted.get('Description')
            ori_desc = origin.get('Description')

            if new_desc != ori_desc:
                for mode in [Mode.efficiency, Mode.balance, Mode.performance]:
                    if mode.value in new_desc:
                        collection = origin['In Collection'].lower()
                        if collection not in TOOLS[task]:
                            TOOLS[task][collection] = dict()
                        TOOLS[task][collection][mode.name] = ToolMeta(
                            model=origin['Name'],
                            description=new_desc,
                            mode=mode,
                            tool_type=tool_type)
                        break

            if new_desc is None:
                new_desc = f'Tool used for {task}'
            
            TOOLS[task][formatted['Name']] = ToolMeta(
                model=origin['Name'],
                description=new_desc,
                tool_type=tool_type)

    for repo in REPOS:
        spec = find_spec(repo)
        if spec is None:
            warnings.warn(f'Local tools from {repo} will not be loaded')
            continue
        package_path = spec.submodule_search_locations[0]

        # Load model-index from source code
        if osp.exists(osp.join(osp.dirname(package_path), 'model-index.yml')):
            model_index_path = osp.join(
                osp.dirname(package_path),'model-index.yml')
        # Load model-index from .mim in installed package.
        else:
            mim_path = osp.join(package_path, '.mim')
            model_index_path = osp.join(
                mim_path, 'model-index.yml')
        model_index: ModelIndex = modelindex.load(model_index_path)
        model_index.build_models_with_collections()

        all_info = defaultdict(list)
        model: Model
        for model in model_index.models:
            model = model.full_model
            all_info[model.in_collection].append(model)

        for _, models in all_info.items():
            # `models_by_data` store models with the same task and evaluation
            # data in the same key. just like:
            # {'ImageNet-1k::Image Classification': [model1, model2, ...]}
            models_by_data = defaultdict(list)
            for model in models:
                # selfsup model without Results should not
                # be considered as a tool
                if model.results is None:
                    continue

                for result in model.results:
                    dataset_task = f'{result.dataset}::{result.task}'
                    model = _model_index_to_dict(model)
                    model['Result'] = result.data
                    models_by_data[dataset_task].append(model)
            
            for dataset_task, models in models_by_data.items():
                _, task = dataset_task.split('::')
                task = task.lower()
                tool_type: BaseTool = task2tool[task]
                info = list(model for model in models)
                formatted_info = tool_type.format_description(info)
                _build_meta_from_formatted_info(info, formatted_info, task)
