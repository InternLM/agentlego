# Copyright (c) OpenMMLab. All rights reserved.
import re
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseParser(metaclass=ABCMeta):

    @abstractmethod
    def parse_inputs(self, inputs: Any, input_types: tuple[str]) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def parse_outputs(self, outputs: Any, output_types: tuple[str]) -> Any:
        raise NotImplementedError

    def description_to_input_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{input:[ ]*(.*?)[ ]*}}}', description))

    def description_to_output_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{output:[ ]*(.*?)[ ]*}}}', description))

    def refine_description(self, description: str) -> str:
        return description
