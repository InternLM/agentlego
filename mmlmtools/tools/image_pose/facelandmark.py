# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser


class HumanFaceLandmark(BaseTool):
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
        try:
            from mmlmtools.tools.image_pose.image_to_pose import \
                load_mmpose_inferencer
        except ImportError as e:
            raise ImportError(f'Failed to run the tool for {e}')

        self._inferencer = load_mmpose_inferencer(
            self.toolmeta.model['pose2d'], self.device)

    def apply(self, image: str) -> str:
        if self.remote:
            raise NotImplementedError
        else:
            output_path = get_new_file_path(image, func_name='face-landmark')
            next(
                self._inferencer(
                    inputs=image,
                    vis_out_dir=output_path,
                    skeleton_style='mmpose',
                ))
        return output_path
