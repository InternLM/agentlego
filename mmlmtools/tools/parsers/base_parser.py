# Copyright (c) OpenMMLab. All rights reserved.
import re
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseParser(metaclass=ABCMeta):

    @abstractmethod
    def parse_inputs(self, inputs: Any) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def parse_outputs(self, outputs: Any) -> Any:
        raise NotImplementedError

    def bind_tool(self, tool: Any) -> None:
        pass

    def description_to_input_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{\s*input:\s*(.*?)\s*}}}', description))

    def description_to_output_types(self, description: str) -> tuple[str]:
        return tuple(re.findall(r'{{{\s*output:\s*(.*?)\s*}}}', description))

    def refine_description(self, description: str) -> str:
        return description
