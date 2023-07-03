# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseParser(metaclass=ABCMeta):

    @abstractmethod
    def parser_input(self, inputs) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def parser_output(self, outputs) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def refine_description(self, description: str) -> str:
        raise NotImplementedError
