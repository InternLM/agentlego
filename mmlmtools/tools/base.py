# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

import torch

from mmlmtools.schema import ToolMeta


class BaseTool(metaclass=ABCMeta):

    def __init__(self, toolmeta: Union[dict, ToolMeta], parser: Callable):
        if isinstance(toolmeta, dict):
            toolmeta = ToolMeta(**toolmeta)
        self.toolmeta = toolmeta
        self.parser = parser(self)
        self._is_setup = False

    @property
    def name(self) -> str:
        return self.toolmeta.name

    def set_parser(self, parser: Callable):
        self.parser = parser(self)

    @property
    def description(self) -> str:
        return self.parser.refine_description()

    def setup(self):
        """Implement lazy initialization here that will be performed before the
        first call of ```apply()```, for example loading the model."""
        self._is_setup = True

    def __call__(self, *args: Any, **kwargs) -> Any:

        if not self._is_setup:
            self.setup()
            self._is_setup = True

        inputs, kwinputs = self.parser.parse_inputs(*args, **kwargs)
        with torch.no_grad():
            outputs = self.apply(*inputs, **kwinputs)
        results = self.parser.parse_outputs(outputs)
        return results

    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def __repr__(self) -> str:
        repr_str = (f'{type(self).__name__}('
                    f'toolmeta={self.toolmeta}, '
                    f'parser={type(self.parser).__name__})')
        return repr_str

    @property
    def input_fields(self):
        return [
            name for name in inspect.getfullargspec(self.apply).args
            if name != 'self'
        ]
