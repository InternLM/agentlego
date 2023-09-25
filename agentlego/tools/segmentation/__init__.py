# Copyright (c) OpenMMLab. All rights reserved.
from .segment_anything import ObjectSegmenting, SegmentAnything, SegmentClicked
from .semantic_segmentation import SemanticSegmentation

__all__ = [
    'SegmentAnything', 'SegmentClicked', 'ObjectSegmenting',
    'SemanticSegmentation'
]
