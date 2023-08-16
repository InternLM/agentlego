# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os.path as osp
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import cv2
import numpy as np
import torch
from PIL import Image

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.toolmeta import ToolMeta
from .base_parser import BaseParser
from .utils import Audio


class converter():

    def __get__(self, instance: Any, owner: Any) -> Callable:
        if instance is None:
            return self.func
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)

    def __init__(self, category: str, source_type: str, target_type: str):
        self.category = category
        self.source_type = source_type
        self.target_type = target_type

    def __call__(self, func: Callable):
        self.func = func
        return self

    def __set_name__(self, owner: Any, name: str) -> None:
        if not hasattr(owner, '_converters'):
            setattr(owner, '_converters', {})

        converters = getattr(owner, '_converters')

        key = (self.category, self.source_type, self.target_type)
        if key in converters:
            raise ValueError(
                f'Duplicated data converter for category `{self.category}` '
                f'from `{self.source_type}` to `{self.target_type}`.')

        converters[key] = name


@dataclass
class ToolInputInfo:
    name: str
    required: bool


class TypeMappingParser(BaseParser):
    # mapping from data category to data type on the agent side
    # e.g. {'image': 'path', 'text': 'string'}
    _agent_cat2type: Dict[str, str]

    # default mapping from data category to data type on the agent side
    _default_agent_cat2type: Dict[str, str] = {}

    # mapping from data mode (category, source_type, target_type) to converter
    # function name. Converters should be instance method decorated by
    # `@converter`.
    _converters: Dict[Tuple[str, str, str], str]

    # mapping from tool argument (i.e. the argument of `apply` method) type
    # to data type for each data category
    _toolarg2type: Dict[str, Dict[Type, str]] = {
        'image': {
            str: 'path',
            Image.Image: 'pillow',
            np.ndarray: 'ndarray'
        },
        'text': {
            str: 'string',
        },
        'audio': {
            str: 'path',
            Audio: 'audio',
        },
    }
    _file_suffix = {
        'image': 'jpg',
        'audio': 'wav',
    }

    def __init__(self, agent_datacat2type: Optional[Dict[str, str]] = None):

        if agent_datacat2type is not None:
            self._agent_cat2type = agent_datacat2type.copy()
        else:
            self._agent_cat2type = self._default_agent_cat2type.copy()

        # sanity check for `self._agent_cat2type`
        for cat, t in self._agent_cat2type.items():
            if cat not in self._toolarg2type:
                raise ValueError(
                    f'The data category `{cat}` is not supported on the tool '
                    'side. (Supported categories: '
                    f'{self._toolarg2type.keys()}))')

            if t not in self._toolarg2type[cat].values():
                raise ValueError(
                    f'The data type `{t}` for category `{cat}` is '
                    'not supported on the tool side. (Supported types: '
                    f'{self._toolarg2type[cat].values()})')

        # The input/output converters and tool argument information will be
        # determined when bound to a tool
        self._input_converters: Optional[list[Callable]] = None
        self._output_converters: Optional[list[Callable]] = None
        self._input_info: Optional[list[ToolInputInfo]] = None

    def bind_tool(self, tool: Any) -> None:
        assert hasattr(tool, 'apply') and callable(tool.apply)
        assert hasattr(tool, 'toolmeta') and isinstance(
            tool.toolmeta, ToolMeta)

        input_cats = tool.inputs
        output_cats = tool.outputs

        agent_input_types = [self._agent_cat2type[c] for c in input_cats]
        agent_output_types = [self._agent_cat2type[c] for c in output_cats]
        tool_input_types = []
        tool_output_types = []

        # parser tool input types
        tool_argspec = inspect.getfullargspec(tool.apply)
        if tool_argspec.kwonlyargs:
            raise ValueError('The `apply` method of the tool '
                             f'`{tool.name}` should not have keyword-only '
                             'arguments.')

        if len(tool_argspec.args) != len(input_cats) + 1:
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` should have '
                f'{len(input_cats)} argument(s) (excluding `self`) indicated'
                f' by the description, but got {len(tool_argspec.args) - 1} ')

        for c, arg in zip(input_cats, tool_argspec.args[1:]):
            argtype = tool_argspec.annotations.get(arg, None)
            if argtype is None:
                raise ValueError(
                    f'Argument `{arg}` of the `apply` method of the tool '
                    f'`{tool.name}` should have type annotation.')

            input_type = self._toolarg2type[c].get(argtype, None)
            if input_type is None:
                raise ValueError(
                    f'Argument `{arg}` of the `apply` method of the tool '
                    f'`{tool.name}` has type annotation `{argtype}`, '
                    f'which is not supported for data category `{c}`.')

            tool_input_types.append(input_type)

        # parse tool output formats
        if 'return' not in tool_argspec.annotations:
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` should have '
                f'return type annotation. '
                'e.g. `def apply(self, input: str) -> str: ...`')

        returns = tool_argspec.annotations['return']
        if not isinstance(returns, tuple):
            returns = (returns, )

        if len(returns) != len(output_cats):
            raise ValueError(
                f'The `apply` method of the tool `{tool.name}` '
                f'should have {len(output_cats)} return(s) indicated '
                f'by the description, but got {len(returns)}.')

        for i, (c, rettype) in enumerate(zip(output_cats, returns)):
            output_type = self._toolarg2type[c].get(rettype, None)
            if output_type is None:
                raise ValueError(f'The {i}-th return of the `apply` method of '
                                 f'the tool `{tool.name}` has type '
                                 f'annotation `{rettype}`, which is not '
                                 f'supported for data category {c}.')

            tool_output_types.append(output_type)

        # assign formatting functions to each input/output
        self._input_converters = []
        for c, src_t, tgt_t in zip(input_cats, agent_input_types,
                                   tool_input_types):
            if src_t == tgt_t:
                self._input_converters.append(lambda x: x)
            else:
                if (c, src_t, tgt_t) not in self._converters:
                    raise ValueError(
                        f'No converter for input category `{c}` from '
                        f'`{src_t}` to `{tgt_t}`, required by tool '
                        f'`{tool.name}`.')
                self._input_converters.append(
                    getattr(self, self._converters[(c, src_t, tgt_t)]))

        self._output_converters = []
        for c, src_t, tgt_t in zip(output_cats, tool_output_types,
                                   agent_output_types):
            if src_t == tgt_t:
                self._output_converters.append(lambda x: x)
            else:
                if (c, src_t, tgt_t) not in self._converters:
                    raise ValueError(
                        f'No converter for output category `{c}` from '
                        f'`{src_t}` to `{tgt_t}`, required by tool '
                        f'`{tool.name}`.')
                self._output_converters.append(
                    getattr(self, self._converters[(c, src_t, tgt_t)]))

        # record necessary information (input name, required, etc) to help
        # process kwargs inputs
        self._input_info = [
            ToolInputInfo(name=arg, required=True)
            for arg in tool_argspec.args[1:]
        ]

        if tool_argspec.defaults is not None:
            # mark optional inputs
            for i in range(len(tool_argspec.defaults)):
                self._input_info[-1 - i].required = False

    def parse_inputs(self, *args, **kwargs) -> Tuple[Tuple, Dict]:
        if self._input_converters is None or self._input_info is None:
            raise RuntimeError('The parser is not bound to a tool yet.')

        inputs = tuple(
            converter(input)
            for input, converter in zip(args, self._input_converters))

        kwinputs = {}
        if len(inputs) < len(self._input_converters):
            # process kwargs inputs
            for i, info in enumerate(
                    self._input_info[len(inputs):], start=len(inputs)):
                if info.name in kwargs:
                    # process kwinput with corresponding converter
                    kwinputs[info.name] = self._input_converters[i](
                        kwargs[info.name])
                elif info.required:
                    # raise error if required kwinput is missing
                    raise ValueError(
                        f'Required input `{info.name}` is missing.')

        return inputs, kwinputs

    def parse_outputs(self, outputs: Any) -> Any:
        if self._output_converters is None:
            raise RuntimeError('The parser is not bound to a tool yet.')

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        if len(outputs) != len(self._output_converters):
            raise ValueError(f'Expect {len(self._output_converters)} outputs, '
                             f'but got {len(outputs)}.')

        outputs = tuple(
            converter(output)
            for output, converter in zip(outputs, self._output_converters))

        return outputs[0] if len(outputs) == 1 else outputs

    def refine_description(self, description: str) -> str:

        def _reformat(match: re.Match) -> str:
            data_cat = match.group(2).strip()
            if data_cat not in self._agent_cat2type:
                raise ValueError
            data_type = self._agent_cat2type[data_cat]

            return f'{data_cat} represented by {data_type}'

        return re.sub(r'{{{(input|output):\s*(.*?)}}}', _reformat, description)

    def description_to_inputs(self, description: str) -> Tuple[str]:
        inputs = tuple(re.findall(r'{{{input:\s*(.*?)\s*}}}', description))
        for cat in inputs:
            if cat not in self._toolarg2type:
                raise ValueError(f'Unknown input data category `{cat}`')
        return inputs

    def description_to_outputs(self, description: str) -> Tuple[str]:
        outputs = tuple(re.findall(r'{{{output:\s*(.*?)\s*}}}', description))
        for cat in outputs:
            if cat not in self._toolarg2type:
                raise ValueError(f'Unknown output data category `{cat}`')
        return outputs

    @converter(category='image', source_type='path', target_type='pillow')
    def _image_path_to_pil(self, path: str) -> Image.Image:
        return Image.open(path)

    @converter(category='image', source_type='pillow', target_type='path')
    def _image_pil_to_path(self, image: Image.Image) -> str:
        path = get_new_file_path(
            osp.join('data', 'image', f'temp.{self._file_suffix["image"]}'),
            func_name='_image_pil_to_path')
        image.save(path)
        return path

    @converter(category='image', source_type='pillow', target_type='ndarray')
    def _image_pil_to_ndarray(self, image: Image.Image) -> np.ndarray:
        return np.array(image)

    @converter(category='image', source_type='ndarray', target_type='pillow')
    def _image_ndarray_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    @converter(category='image', source_type='ndarray', target_type='path')
    def _image_ndarray_to_path(self, image: np.ndarray) -> str:
        path = get_new_file_path(
            osp.join('data', 'imag', f'temp.{self._file_suffix["image"]}'),
            func_name='_image_ndarray_to_path')
        cv2.imwrite(path, image)
        return path

    @converter(category='image', source_type='path', target_type='ndarray')
    def _image_path_to_ndarray(self, path: str) -> np.ndarray:
        return cv2.imread(path)

    @converter(category='audio', source_type='path', target_type='audio')
    def _audio_path_to_audio(self, path: str) -> Audio:
        return Audio.from_path(path)

    @converter(category='audio', source_type='audio', target_type='path')
    def _audio_to_audio_path(self, audio: Union[Audio, str]) -> str:
        if isinstance(audio, str):
            return str
        # TODO: Only support audio with one channel: [1, N] now.
        try:
            import torchaudio
        except ImportError as e:
            raise ImportError(f'Failed to run the tool: {e} '
                              '`torchaudio` is not installed correctly')
        saved_path = get_new_file_path(
            osp.join('data', 'audio', f'temp.{self._file_suffix["audio"]}'),
            func_name='_ndarray_to_audio_path')
        torchaudio.save(saved_path,
                        torch.from_numpy(audio.array).reshape(1, -1).float(),
                        audio.sampling_rate)
        return saved_path
