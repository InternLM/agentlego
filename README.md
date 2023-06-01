# OpenMMLab Visual Toolbox for LLMs

## Visual ChatGPT

```Python
from mmlmtools import list_tool, load_tool

tools = []
models = {}

mmtools = list_tool()  # get the list of mmtools
# dict_keys(['ImageCaptionTool', 'Text2BoxTool', 'Text2ImageTool', 'OCRTool'])

for tool_name in mmtools:
    # obtain tool instance and toolmeta via `load_tool()`
    mmtool, toolmeta = load_tool(tool_name, device='cpu')

    models[tool_name] = mmtool
    tools.append(
        Tool(
            name=toolmeta.tool_name,
            description=toolmeta.description,
            func=mmtool.apply))
```

## 添加新工具

### 1. 创建文件

- 在 tools/ 目录下新建对应repo的文件，例如：mmdet.py
- Tool 命名要能体现功能，可以参考 Inferencer 命名，例如：Text2ImageTool, OCRTool
- 新的工具必须继承基类 BaseTool

```Python
from .base_tool import BaseTool

class Text2BoxTool(BaseTool):
    ...
```

### 2. 重载两个 convert （可选）

- convert_inputs 用于把 LLM 传给 Tool 的内容解析成推理接口需要的格式，例如：
  - GLIP 的推理接口为 self.inferencer(imgs=image_path, text_prompt=text)
  - convert_inputs 把 '1.jpg, where is the tree?' 解析成

```Python
def convert_inputs(self, inputs, **kwargs):
    image_path, text = inputs.split(',')
    return image_path, text
```

默认情况下 convert_inputs 和 convert_outputs 都会直接 return inputs 和 return outputs

### 3. 实现 infer

最重要的是实现 infer，infer是整个工具推理的核心代码。

对于ImageCaption工具而言，输入输出都是文本，所以实现比较简单，大家注意 scope 的切换就好

```python
def infer(self, inputs, **kwargs):
    if self.remote:
        raise NotImplementedError
    else:
        with Registry('scope').switch_scope_and_registry('mmpretrain'):
            outputs = self.inferencer(inputs)[0]['pred_caption']
    return outputs
```

对于检测工具 GLIP 而言，输入有两个：image_path 和 text，在convert_inputs已经完成了解析

```Python
def infer(self, inputs, **kwargs):
    image_path, text = inputs
    if self.remote:
        ...
    else:
        with Registry('scope').switch_scope_and_registry('mmdet'):
                results = self.inferencer(imgs=image_path, text_prompt=text)
                output_path = get_new_image_name(
                    image_path, func_name='detect-something')
                img = mmcv.imread(image_path)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                self.visualizer.add_datasample(
                    'results',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    show=False,
                    wait_time=0,
                    out_file=output_path,
                    pred_score_thr=0.5)

        return output_path
```

在这里例子中，我们假设Tool的输出也是 image_path ，因此在 infer() 中直接调用了 visualizer 来把图片存到本地。

但是假如我们的 Tool 想要适配不同的系统，Tool 的输出就需要在 convert_outputs 中进行转码，可以转成 image_path 也可以转成 Tensor 或者别的特定格式

### 4. 加入到 DEFAULT_TOOLS

对于 MM 系列的工具而言，需要默认加入到 api.py 下的 DEFAULT_TOOLS
格式为：

```python
'类名': dict(
    model='传给inferencer的模型初始化key',
    description='写给LLM的工具描述'
)
```

例如：

```python
DEFAULT_TOOLS = {
    'ImageCaptionTool':
    dict(
        model='blip-base_3rdparty_caption',
        description=
        'useful when you want to know what is inside the photo. receives image_path as input. The input to this tool should be a string, representing the image_path. '  # noqa
    ),
    'Text2BoxTool':
    dict(
        model='glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
        description=
        'useful when you only want to detect or find out given objects in the picture. The input to this tool should be a comma separated string of two, representing the image_path, the text description of the object to be found'  # noqa
    ),
}
```
