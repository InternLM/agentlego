# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseToolModule(metaclass=ABCMeta):
    """"""

    # @abstractmethod
    # def convert_inputs(self, inputs, **kwargs):
    #     pass

    # @abstractmethod
    # def convert_outputs(self, outputs, **kwargs):
    #     pass

    @abstractmethod
    def apply(self, inputs, **kwargs):
        pass
