# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry
from mmpretrain.apis import ImageCaptionInferencer

from mmlmtools.toolmeta import ToolMeta
from .base_tool import BaseTool


class ImageCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        tool_name='ImageCaptionTool',
        model='blip-base_3rdparty_caption',
        description='This is a useful tool '
        'when you want to know what is inside the image.')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda',
                 **init_args):
        super().__init__(toolmeta, input_style, output_style, remote, device,
                         **init_args)

        self.inferencer = None

    def setup(self):
        if self.inferencer is None:
            self.inferencer = ImageCaptionInferencer(
                self.toolmeta.model, device=self.device, **self.init_args)

    def apply(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmpretrain'):
                outputs = self.inferencer(inputs, **kwargs)[0]['pred_caption']
        return outputs
