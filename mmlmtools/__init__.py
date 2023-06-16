# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .api import custom_tool, list_tools, load_tool

try:
    import mmagic  # noqa
except Exception:
    warnings.warn('Import mmagic failed. '
                  'Please check whether it is installed properly.')

try:
    import mmdet  # noqa
except Exception:
    warnings.warn('Import mmdet failed. '
                  'Please check whether it is installed properly.')

try:
    import mmocr  # noqa
except Exception:
    warnings.warn('Import mmocr failed. '
                  'Please check whether it is installed properly.')

try:
    import mmpose  # noqa
except Exception:
    warnings.warn('Import mmpose failed. '
                  'Please check whether it is installed properly.')

try:
    import mmpretrain  # noqa
except Exception:
    warnings.warn('Import mmpretrain failed. '
                  'Please check whether it is installed properly.')

__all__ = ['load_tool', 'custom_tool', 'list_tools']
