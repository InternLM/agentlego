# Copyright (c) OpenMMLab. All rights reserved.
from .image_canny import CannyTextToImage, ImageToCanny
from .image_depth import DepthTextToImage, ImageToDepth
from .image_editing import (ImageExpansion, ImageStylization, ObjectRemove,
                            ObjectReplace)
from .image_pose import HumanBodyPose, HumanFaceLandmark, PoseToImage
from .image_scribble import ImageToScribble, ScribbleTextToImage
from .image_text import ImageCaption, TextToImage
from .imagebind import (AudioImageToImage, AudioTextToImage, AudioToImage,
                        ThermalToImage)
from .object_detection import ObjectDetection, TextToBbox
from .ocr import OCR
from .segmentation import SegmentAnything, SegmentObject, SemanticSegmentation
from .speech_text import SpeechToText, TextToSpeech
from .translation import Translation
from .vqa import VisualQuestionAnswering

__all__ = [
    'CannyTextToImage', 'ImageToCanny', 'DepthTextToImage', 'ImageToDepth',
    'ImageExpansion', 'ObjectRemove', 'ObjectReplace', 'HumanFaceLandmark',
    'HumanBodyPose', 'PoseToImage', 'ImageToScribble', 'ScribbleTextToImage',
    'ImageCaption', 'TextToImage', 'VisualQuestionAnswering',
    'ObjectDetection', 'TextToBbox', 'OCR', 'SegmentObject', 'SegmentAnything',
    'SemanticSegmentation', 'ImageStylization', 'AudioToImage',
    'ThermalToImage', 'AudioImageToImage', 'AudioTextToImage', 'SpeechToText',
    'TextToSpeech', 'Translation'
]
