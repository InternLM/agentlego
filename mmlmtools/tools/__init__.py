# Copyright (c) OpenMMLab. All rights reserved.
from .image_canny.canny_to_image import CannyTextToImage
from .image_canny.image_to_canny import ImageToCanny
from .image_depth.depth_to_image import DepthTextToImage
from .image_depth.image_to_depth import ImageToDepth
from .image_editing.extension import ImageExtension
from .image_editing.remove import ObjectRemove
from .image_editing.replace import ObjectReplace
from .image_editing.stylization import InstructPix2Pix
from .image_pose.facelandmark import HumanFaceLandmark
from .image_pose.image_to_pose import HumanBodyPose
from .image_pose.pose_to_image import PoseToImage
from .image_scribble.image_to_scribble import ImageToScribble
from .image_scribble.scribble_to_image import ScribbleTextToImage
from .image_text.image_to_text import ImageCaption
from .image_text.text_to_image import TextToImage
from .imagebind import (AudioImageToImage, AudioTextToImage, AudioToImage,
                        ThermalToImage,)
from .object_detection.object_detection import ObjectDetection
from .object_detection.text_to_bbox import TextToBox
from .ocr.ocr import OCR, ImageMaskOCR
from .segmentation.segment_anything import (ObjectSegmenting, SegmentAnything,
                                            SegmentClicked,)
from .segmentation.semantic_segmentation import SemanticSegmentation
from .text_qa import TextQuestionAnswering
from .text_to_speech import TextToSpeechTool
from .vqa.visual_question_answering import VisualQuestionAnswering

__all__ = [
    'CannyTextToImage', 'ImageToCanny', 'DepthTextToImage', 'ImageToDepth',
    'ImageExtension', 'ObjectRemove', 'ObjectReplace', 'HumanFaceLandmark',
    'HumanBodyPose', 'PoseToImage', 'ImageToScribble', 'ScribbleTextToImage',
    'ImageCaption', 'TextToImage', 'VisualQuestionAnswering',
    'ObjectDetection', 'TextToBox', 'ImageMaskOCR', 'OCR', 'ObjectSegmenting',
    'SegmentAnything', 'SegmentClicked', 'SemanticSegmentation',
    'InstructPix2Pix', 'TextQuestionAnswering', 'AudioToImage',
    'ThermalToImage', 'AudioImageToImage', 'AudioTextToImage',
    'TextQuestionAnswering', 'TextToSpeechTool'
]
