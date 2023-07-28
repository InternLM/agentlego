# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from mmlmtools.toolmeta import ToolMeta
from .parsers import BaseParser, NaiveParser


class BaseTool(metaclass=ABCMeta):
    DEFAULT_TOOLMETA = dict(
        name='Abstract Base Tool',
        model=None,
        description='This is an abstract tool interface. '
        'A tool class should inherit from this class and specify the '
        'inputs (e.g. {{{input: image}}}) and the outputs '
        '(e.g. {{{output: text}}}) in its description.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cpu'):
        self.remote = remote
        self.device = device
        self._is_setup = False

        if toolmeta is not None:
            self.toolmeta = toolmeta
        else:
            assert hasattr(self, 'DEFAULT_TOOLMETA')
            assert self.DEFAULT_TOOLMETA.get('name', None) is not None, (
                '`name` in `DEFAULT_TOOLMETA` should not be None.')
            assert self.DEFAULT_TOOLMETA.get(
                'description', None) is not None, (
                    '`description` in `DEFAULT_TOOLMETA` should not be None.')
            self.toolmeta = ToolMeta(**self.DEFAULT_TOOLMETA)

        self.parser = parser if parser is not None else NaiveParser()

        # Parse data categories of inputs and outputs from description
        # if not specified in toolmeta. This should be done before
        # binding a parser to the tool.
        if self.toolmeta.inputs is None:
            self.inputs = self.parser.description_to_inputs(
                self.toolmeta.description)
        else:
            self.inputs = self.toolmeta.inputs

        if self.toolmeta.outputs is None:
            self.outputs = self.parser.description_to_outputs(
                self.toolmeta.description)
        else:
            self.outputs = self.toolmeta.outputs

        self.parser.bind_tool(self)

    @property
    def name(self) -> str:
        return self.toolmeta.name

    @property
    def description(self) -> str:
        return self.parser.refine_description(self.toolmeta.description)

    def setup(self):
        """Implement lazy initialization here that will be performed before the
        first call of ```apply()```, for example loading the model."""
        pass

    def __call__(self, *args: Any, **kwargs) -> Any:

        if not self._is_setup:
            self.setup()
            self._is_setup = True

        inputs, kwinputs = self.parser.parse_inputs(*args, **kwargs)
        outputs = self.apply(*inputs, **kwinputs)
        results = self.parser.parse_outputs(outputs)
        return results

    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError
