# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmpose.apis import MMPoseInferencer

from mmlmtools.toolmeta import ToolMeta
from mmlmtools.utils import get_new_image_name
from .base_tool import BaseTool
from .parsers import BaseParser


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
            self._inferencer = MMPoseInferencer(
                pose2d=self.toolmeta.model['pose2d'], device=self.device)

    def apply(self, image_path: str) -> str:
        output_path = get_new_image_name(
            image_path, func_name='pose-estimation')
        if self.remote:
            raise NotImplementedError
        else:
            next(
                self._inferencer(
                    image_path=image_path,
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
        self._inferencer = MMPoseInferencer(
            pose2d=self.toolmeta.model['pose2d'], device=self.device)

    def apply(self, image_path: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_image_name(
                image_path, func_name='face-landmark')
            next(
                self._inferencer(
                    image_path=image_path,
                    vis_out_dir=output_path,
                    skeleton_style='mmpose',
                ))
        return output_path
