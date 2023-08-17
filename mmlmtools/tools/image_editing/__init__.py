# Copyright (c) OpenMMLab. All rights reserved.

from .extension import ImageExtension
from .remove import ObjectRemove
from .replace import ObjectReplace
from .stylization import InstructPix2PixTool

__all__ = [
    'ImageExtension', 'ObjectRemove', 'ObjectReplace', 'InstructPix2PixTool'
]
