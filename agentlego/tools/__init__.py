from .base import BaseTool
from .calculator import Calculator
from .func import make_tool
from .image_canny import CannyTextToImage, ImageToCanny, ReplaceBackgroundOrForeground
from .image_depth import DepthTextToImage, ImageToDepth
from .image_editing import (AddText, DrawBox, ImageExpansion, ImageStylization,
                            ObjectRemove, ObjectReplace)
from .image_pose import HumanBodyPose, HumanFaceLandmark, PoseToImage
from .image_scribble import ImageToScribble, ScribbleTextToImage
from .image_text import ImageDescription, ImageRegionDescription, TextToImage
from .imagebind import AudioImageToImage, AudioTextToImage, AudioToImage, ThermalToImage
from .object_detection import (CelebrityRecognition, CountGivenObject, ObjectDetection,
                               TextToBbox)
from .ocr import OCR, MathOCR
from .python_interpreter import Plot, PythonInterpreter, Solver
from .search import GoogleSearch
from .segmentation import SegmentAnything, SegmentObject, SemanticSegmentation
from .speech_text import SpeechToText, TextToSpeech
from .translation import Translation
from .vqa import VQA

__all__ = [
    'CannyTextToImage', 'ImageToCanny', 'DepthTextToImage', 'ImageToDepth',
    'ImageExpansion', 'ObjectRemove', 'ObjectReplace', 'HumanFaceLandmark',
    'HumanBodyPose', 'PoseToImage', 'ImageToScribble', 'ScribbleTextToImage',
    'ImageDescription', 'TextToImage', 'VQA', 'ObjectDetection', 'TextToBbox', 'OCR',
    'SegmentObject', 'SegmentAnything', 'SemanticSegmentation', 'ImageStylization',
    'AudioToImage', 'ThermalToImage', 'AudioImageToImage', 'AudioTextToImage',
    'SpeechToText', 'TextToSpeech', 'Translation', 'GoogleSearch', 'Calculator',
    'BaseTool', 'make_tool', 'AddText', 'DrawBox', 'ImageRegionDescription',
    'CountGivenObject', 'ReplaceBackgroundOrForeground', 'MathOCR', 'PythonInterpreter',
    'Plot', 'Solver', 'CelebrityRecognition'
]
