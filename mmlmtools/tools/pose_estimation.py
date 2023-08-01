# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmpose.apis import MMPoseInferencer

from mmlmtools.utils import get_new_image_path
from mmlmtools.utils.cached_dict import CACHED_TOOLS
from mmlmtools.utils.toolmeta import ToolMeta
from .base_tool import BaseTool
from .parsers import BaseParser


def load_pose_inferencer(model, device):
    if CACHED_TOOLS.get('pose_inferencer', None) is not None:
        pose_inferencer = CACHED_TOOLS['pose_inferencer'][model]
    else:
        pose_inferencer = MMPoseInferencer(pose2d=model, device=device)
        CACHED_TOOLS['pose_inferencer'][model] = pose_inferencer
    return pose_inferencer


def load_facelandmark_inferencer(model, device):
    if CACHED_TOOLS.get('facelandmark_inferencer', None) is not None:
        facelandmark_inferencer = CACHED_TOOLS['facelandmark_inferencer'][
            model]
    else:
        facelandmark_inferencer = MMPoseInferencer(pose2d=model, device=device)
        CACHED_TOOLS['facelandmark_inferencer'][
            model] = facelandmark_inferencer
    return facelandmark_inferencer


class HumanBodyPoseTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Human Body Pose Detection On Image',
        model={'pose2d': 'human'},
        description='This is a useful tool when you want to draw '
        'or show the skeleton of human, or estimate the pose or keypoints '
        'of human in a photo. It takes an {{{input:image}}} as the input, '
        'and returns a {{{output:image}}} representing the image with '
        'skeletons. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        if self.remote:
            from mmpose.datasets.datasets.utils import parse_pose_metainfo
            from mmpose.registry import DATASETS, VISUALIZERS
            self._inferencer = True
            visualizer_cfg = {
                'type': 'PoseLocalVisualizer',
                'vis_backends': [{
                    'type': 'LocalVisBackend'
                }],
                'name': 'visualzier',
                '_scope_': 'mmpose',
                'radius': 3,
                'alpha': 0.8,
                'line_width': 1,
            }
            metainfo = DATASETS.get('CocoDataset').METAINFO
            dataset_meta = parse_pose_metainfo(metainfo)
            self.visualizer = VISUALIZERS.build(visualizer_cfg)
            self.visualizer.set_dataset_meta(
                dataset_meta, skeleton_style='openpose')
        else:
            self._inferencer = load_pose_inferencer(
                self.toolmeta.model['pose2d'], self.device)

    def apply(self, image: str) -> str:
        output_path = get_new_image_path(image, func_name='pose-estimation')
        if self.remote:
            raise NotImplementedError
        else:
            next(
                self._inferencer(
                    inputs=image,
                    vis_out_dir=output_path,
                    skeleton_style='openpose',
                ))
        return output_path


class HumanFaceLandmarkTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Human Face Landmark On Image',
        model={'pose2d': 'face'},
        description='This is a useful tool when you want to draw or show the '
        'landmark of human faces, or estimate the keypoints of human face in '
        'a photo. It takes an {{{input:image}}} as the input, and returns a '
        '{{{output:image}}} representing the image with landmarks. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_facelandmark_inferencer(
            self.toolmeta.model['pose2d'], self.device)

    def apply(self, image: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_path(image, func_name='face-landmark')
            next(
                self._inferencer(
                    inputs=image,
                    vis_out_dir=output_path,
                    skeleton_style='mmpose',
                ))
        return output_path
