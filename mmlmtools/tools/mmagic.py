# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.api import MMagicInferencer

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

        self.image_path = './output/sd_res.png'
        self.inferencer = MMagicInferencer(
            model_name=model, device=device, **kwargs)

    def inference(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            inputs = self.inferencer.preprocess(text=inputs)
            outputs = self.inferencer(inputs)
        return outputs

    def convert_outputs(self, outputs, **kwargs):
        if self.output_style == 'image_path':
            self.inferencer.visualize(
                preds=outputs, result_out_dir=self.image_path)
            return self.image_path
        elif self.output_style == 'image':
            return outputs
        else:
            raise NotImplementedError
