# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.apis import VisualQuestionAnsweringInferencer

from mmlmtools.toolmeta import ToolMeta
from .base_tool import BaseTool


class VisualQuestionAnsweringTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Visual Question Answering',
        model={'model': 'ofa-base_3rdparty-zeroshot_vqa'},
        description='This is a useful tool '
        'when you want to know some information about the image.'
        'you can ask questions like "what is the color of the car?"',
        input_description='The input to this tool should be a comma separated '
        'string of two, representing the image path and the question.',
        output_description='It returns a string as the output, '
        'representing the answer to the question. ')

    def __init__(self,
                 toolmeta: ToolMeta = None,
                 input_style: str = 'image_path, text',
                 output_style: str = 'text',
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(
            toolmeta,
            input_style,
            output_style,
            remote,
            device
        )

        self._inferencer = None

    def setup(self):
        if self._inferencer is None:
            self._inferencer = VisualQuestionAnsweringInferencer(
                model=self.toolmeta.model['model'], device=self.device)

    def convert_inputs(self, inputs):
        if self.input_style == 'image_path, text':
            split_inputs = inputs.split(',')
            image_path = split_inputs[0]
            text = ','.join(split_inputs[1:])
        return image_path, text

    def apply(self, inputs):
        image_path, text = inputs
        if self.remote:
            raise NotImplementedError
        else:
            outputs = self._inferencer(image_path, text)[0]['pred_answer']
            print(outputs)
        return outputs

    def convert_outputs(self, outputs):
        if self.output_style == 'text':
            outputs = ', '.join(outputs)
            return outputs
        else:
            raise NotImplementedError
