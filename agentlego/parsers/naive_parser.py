from typing import Tuple

from .base_parser import BaseParser


class NaiveParser(BaseParser):

    def parse_inputs(self, *args, **kwargs) -> Tuple[tuple, dict]:
        return args, kwargs

    def parse_outputs(self, outputs):
        return outputs
