# Copyright (c) OpenMMLab. All rights reserved.
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple


class BaseParser(metaclass=ABCMeta):

    @abstractmethod
    def parse_inputs(self, *args, **kwargs) -> Tuple[Tuple, Dict]:
        raise NotImplementedError

    @abstractmethod
    def parse_outputs(self, outputs: Any) -> Any:
        raise NotImplementedError

    def bind_tool(self, tool: Any) -> None:
        pass

    def description_to_inputs(self, description: str) -> Tuple[str]:
        return tuple(re.findall(r'{{{\s*input:\s*(.*?)\s*}}}', description))

    def description_to_outputs(self, description: str) -> Tuple[str]:
        return tuple(re.findall(r'{{{\s*output:\s*(.*?)\s*}}}', description))

    def refine_description(self, description: str) -> str:
        return description
