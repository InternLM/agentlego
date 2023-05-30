# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.apis import MMagicInferencer
from mmengine import Registry

from .base_tool import BaseTool


class Text2ImageTool(BaseTool):

    def __init__(self,
                 model: str = 'stable_diffusion',
                 checkpoint: str = None,
                 input_style: str = 'text',
                 output_style: str = 'image_path',
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        self.a_prompt = 'best quality, extremely detailed'
        self.image_path = './image/sd_res.png'
        self.inferencer = MMagicInferencer(
            model_name=model, device=device, **kwargs)

    def infer(self, inputs, **kwargs):
        inputs += self.a_prompt
        if self.remote:
            raise NotImplementedError
        else:
            with Registry('scope').switch_scope_and_registry('mmagic'):
                self.inferencer.infer(
                    text=inputs, result_out_dir=self.image_path)
        return self.image_path

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'image_path':
            return outputs
        else:
            raise NotImplementedError
