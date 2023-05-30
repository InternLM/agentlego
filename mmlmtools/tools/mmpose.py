# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmpose.apis import MMPoseInferencer

from ..utils.utils import get_new_image_name
from .base_tool import BaseTool


class HumanBodyPoseTool(BaseTool):

    def __init__(self,
                 model: str = 'human',
                 checkpoint: str = None,
                 input_style: str = 'image_path',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.inferencer = MMPoseInferencer(
            model_name=model, device=device, **kwargs)

    def infer(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            image_path = get_new_image_name(
                'image/pose-res.png', func_name='estimate-pose')
            with Registry('scope').switch_scope_and_registry('mmpose'):
                self.inferencer.infer(
                    inputs, vis_out_dir=image_path)
        return image_path

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'image_path':
            return outputs
        else:
            raise NotImplementedError
