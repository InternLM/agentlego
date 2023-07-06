# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from .base_parser import BaseParser


class TypeMappingParser(BaseParser):
    TypeMapping: dict[str, str] = {}

    def __init__(self, type_mapping: Optional[dict[str, str]] = None):

        if type_mapping is not None:
            self.type_mapping = type_mapping.copy()
        else:
            self.type_mapping = self.TypeMapping.copy()
