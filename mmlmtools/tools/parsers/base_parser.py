# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseParser(metaclass=ABCMeta):

    @abstractmethod
    def parse_inputs(self, inputs: Any, input_types: tuple[str]) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def parse_outputs(self, outputs: Any, output_types: tuple[str]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def refine_description(self, description: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def description_to_input_types(self, description: str) -> tuple[str]:
        raise NotImplementedError

    @abstractmethod
    def description_to_output_types(self, description: str) -> tuple[str]:
        raise NotImplementedError
