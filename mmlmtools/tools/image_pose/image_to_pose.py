# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cache import load_or_build_object
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class HumanBodyPose(BaseTool):
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
            from mmpose.apis import MMPoseInferencer
            self._inferencer = load_or_build_object(
                MMPoseInferencer,
                pose2d=self.toolmeta.model['pose2d'],
                device=self.device)

    def apply(self, image: str) -> str:
        output_path = get_new_file_path(image, func_name='pose-estimation')
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
