# Copyright (c) OpenMMLab. All rights reserved.
from .base_parser import BaseParser


class TypeMappingParser(BaseParser):
    TypeMapping: dict[str, str] = {}
