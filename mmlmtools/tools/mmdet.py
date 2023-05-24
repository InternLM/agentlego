# Copyright (c) OpenMMLab. All rights reserved.
from mmdet import DetInferencer

from .base_tool import BaseTool


class DetTool(BaseTool):

    def __init__(self,
                 model: str = None,
                 checkpoint: str = None,
                 input_style: str = None,
                 output_style: str = None,
                 remote: bool = False,
                 device: str = 'cuda',
                 **kwargs):
        super().__init__(model, checkpoint, input_style, output_style, remote,
                         **kwargs)

        # init_args = {
        #     'model': 'rtmdet-s',
        #     'weights': None,
        #     'device': 'cuda:0',
        #     'palette': 'none'
        # }

        self.call_args = {
            'out_dir': 'outputs',
            'pred_score_thr': 0.3,
            'batch_size': 1,
            'no_save_vis': False,
            'no_save_pred': False,
        }
        self.inferencer = DetInferencer(model=model, device=device)

    def inference(self, inputs, **kwargs):
        if self.remote:
            raise NotImplementedError
        else:
            outputs = self.inferencer(inputs=inputs, **self.call_args)
        return outputs
