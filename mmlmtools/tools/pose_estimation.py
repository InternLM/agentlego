# Copyright (c) OpenMMLab. All rights reserved.
# import mmcv
from mmpose.apis import MMPoseInferencer

from mmlmtools.toolmeta import ToolMeta
from ..utils.file import get_new_image_path
from .base_tool_v1 import BaseToolv1


class HumanBodyPoseTool(BaseToolv1):
    DEFAULT_TOOLMETA = dict(
        name='Human Body Pose Detection On Image',
        model={'pose2d': 'human'},
        description='This is a useful tool '
        'when you want to draw or show the skeleton of human, '
        'or estimate the pose or keypoints of human in a photo.',
        input_description='It takes a string as the input, '
        'representing the image_path. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
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

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_path(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        image_path = get_new_image_path(inputs, func_name='pose-estimation')
        if self.remote:
            raise NotImplementedError
            # import json

            # from openxlab.model import inference
            # res = inference('mmpose/human_body', [image_path])
            # # print(f'json result:{json.loads(predict)}')
            # preds = json.loads(res)['predictions'][0][0]
            # keypoints, bbox = preds['keypoints'], preds['bbox']
            # img = mmcv.imread(inputs, channel_order='rgb')
            # self.visualizer.add_datasample(
            #     'result',
            #     img,
            #     data_sample=data_samples,
            #     draw_gt=False,
            #     skeleton_style='openpose',
            #     kpt_thr=0.3,
            #     out_file=image_path
            #     )
        else:

            next(
                self._inferencer(
                    inputs,
                    vis_out_dir=image_path,
                    skeleton_style='openpose',
                ))
        return image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError


class HumanFaceLandmarkTool(BaseToolv1):
    DEFAULT_TOOLMETA = dict(
        name='Human Face Landmark On Image',
        model={'pose2d': 'face'},
        description='This is a useful tool '
        'when you want to draw or show the landmark of human faces, '
        'or estimate the keypoints of human face in a photo.',
        input_description='It takes a string as the input, '
        'representing the image_path. ',
        output_description='It returns a string as the output, '
        'representing the image_path. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, input_style, output_style, remote, device)

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = MMPoseInferencer(
                pose2d=self.toolmeta.model['pose2d'], device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path':  # visual chatgpt style
            return inputs
        elif self.input_style == 'pil image':  # transformer agent style
            temp_image_path = get_new_image_path(
                'image/temp.jpg', func_name='temp')
            inputs.save(temp_image_path)
            return temp_image_path
        else:
            raise NotImplementedError

    def apply(self, inputs):
        if self.remote:
            raise NotImplementedError
        else:
            image_path = get_new_image_path(
                inputs, func_name='pose-estimation')
            next(
                self._inferencer(
                    inputs,
                    vis_out_dir=image_path,
                    skeleton_style='mmpose',
                ))
        return image_path

    def convert_outputs(self, outputs):
        if self.output_style == 'image_path':  # visual chatgpt style
            return outputs
        elif self.output_style == 'pil image':  # transformer agent style
            from PIL import Image
            outputs = Image.open(outputs)
            return outputs
        else:
            raise NotImplementedError
