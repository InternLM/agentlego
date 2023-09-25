# OpenMMLab Visual Toolbox for LLMs

## Visual ChatGPT

### 基础使用

```Python
from agentlego import list_tools, load_tool

tools = []
models = {}

agentlego_tools = list_tools()  # list tools in AgentLego
# dict_keys([
# 'Image2CannyTool',
# 'ImageCaptionTool',
# 'Text2BoxTool',
# 'Text2ImageTool',
# 'OCRTool',
# 'Canny2ImageTool',
# 'ObjectDetectionTool',
# 'HumanBodyPoseTool',
# 'SemSegTool',
# ])

for tool_name in agentlego_tools:
    # obtain tool instance via `load_tool()`
    tool = load_tool(tool_name, device='cpu')

    models[tool_name] = tool
    tools.append(
        Tool(
            name=tool.toolmeta.name,
            description=tool.toolmeta.description,
            func=tool))
```

### 适配已有 Agents

我们提供了方便的接口来适配已有的 Agents，包括：

- 直接返回 langchain Tool
- 将 MMLMTool 转为 Visual ChatGPT / InterGPT 的模型
- 将 MMLMTool 转为 Transformer Agent 的模型

通过两行代码可以快速将 AgentLego 集成到 VisualChatGPT / InterGPT 项目中：

#### VisualChatGPT

```Python
from agentlego.adapter.load_mmtools_for_langchain

# class ConversationBot:
#     def __init__(self, load_dict):
#         # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
#         print(f"Initializing VisualChatGPT, load_dict={load_dict}")
#         if 'ImageCaptioning' not in load_dict:
#             raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

          # Load AgentLego as langchain Tool
          self.tools = load_mmtools_for_langchain(load_dict)

#         self.llm = OpenAI(temperature=0)
#         self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
```

#### InternGPT

```Python
from agentlego.adapter.convert_tools_for_igpt

#class ConversationBot:
#    def __init__(self, load_dict, chat_disabled=False):
#        print(f"Initializing InternGPT, load_dict={load_dict}")
#
#        self.chat_disabled = chat_disabled
#        self.models = {}
#        self.audio_model = whisper.load_model("small").to('cuda:0')
        #self.audio_model = whisper.load_model("small")
        # Load Basic Foundation Models
#        for class_name, device in load_dict.items():
#            self.models[class_name] = globals()[class_name](device=device)

        # Convert MMLMTools into iGPT style
        convert_tools_for_igpt(self.models)

#        # Load Template Foundation Models
#        for class_name, module in globals().items():
#            if getattr(module, 'template_model', False):
#                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
#                loaded_names = set([type(e).__name__ for e in self.models.values()])
#                if template_required_names.issubset(loaded_names):
#                    self.models[class_name] = globals()[class_name](
#                        **{name: self.models[name] for name in template_required_names})
#        self.tools = []
#        for instance in self.models.values():
#            for e in dir(instance):
#                if e.startswith('inference'):
#                    func = getattr(instance, e)
#                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
#
#        print("Current allocated memory:", torch.cuda.memory_allocated())
#        print("models",set([type(e).__name__ for e in self.models.values()]))
```

#### Transformer Agent

```Python
# from transformers import HfAgent
from agentlego.adapter.transformers_agent import load_tools_for_hfagent
tools = load_tools_for_hfagent()
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder",
                additional_tools=tools)
```

### 高级使用

`load_tool()` 允许用户在实例化每个 Tool 时手动修改默认配置：

- `device`: 模型加载的设备
- `name`: 提供给 Agent 的工具名称
- `model`: 推理所使用的模型
- `description`: 工具的功能描述
- `input_description`: 工具的输入格式描述
- `output_description`: 工具的输出格式描述
- `input_style`: Agent 输入到工具的内容格式
- `output_style`: 工具输出给 Agent 的内容格式

```Python

mmtool = load_tool('ImageCaptionTool',
                   device='cuda:0',
                   name='Get Photo Description',
                   description='This is a useful tool '
                               'when you want to know what is inside the image.'
                   input_description='It takes a string as the input, representing the image_path. ',
                   output_description='It returns a text that contains the description of the input image. '
                   )

```

## 添加新工具

### 1. 创建文件

- 在 tools/ 目录下新建对应工具的文件，例如：image_caption.py
- Tool 命名要能体现功能，可以参考 Inferencer 命名，例如：Text2ImageTool, OCRTool
- 新的工具必须继承基类 BaseTool
- 需要定义一个 `dict` 类型的成员 `DEFAULT_TOOLMETA` 作为该工具默认加载的 `ToolMeta`，否则用户必须在实例化这个 Tool 的时候手动定义这些信息（通过 `load_tool(..., model='xxx', description='xxx')`）

```Python
class ImageCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Get Photo Description',
        model='blip-base_3rdparty_caption',
        description='This is a useful tool '
        'when you want to know what is inside the image.')
    ...
```

- `description` 部分只需要提供**功能描述**，不需要写输入输出格式相关的描述。
- 输入输出相关的描述会根据 `self.input_style` 和 `self.output_tyle` 自动生成。
- 最终的工具描述由 `{功能描述} {输入描述} {输出描述}` 拼接而成。
- 你也可以通过添加 `input_description` 和 `output_description` 字段来提供更加精准的输入输出格式描述。

```Python
class ImageCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        ...
        input_description='The input to this tool should be a string, representing the image_path. ',
        output_description='It returns a text that contains the description of the input image. ')
    ...
```

### 2. 定义 setup

定义推理所使用的 inferencer，`setup()` 会在每次 Tool 被调用时运行一次，在首次运行时完成 inferencer 的实例化。

```Python
def setup(self):
    if self._inferencer is None:
        self._inferencer = MMSegInferencer(
            self.toolmeta.model, device=self.device)
```

### 3. 重载两个 convert （可选）

在这里例子中，我们的 `Tool` 输入输出都是 `image_path` ，因此在 `apply()` 中直接调用了 `visualizer` 来把图片存到本地。

但是假如我们的 `Tool` 想要适配不同的系统，`Tool` 的输出就需要在 `convert_inputs` 和 `convert_outputs` 中进行转码，可以转成 `image_path` 也可以转成 `PIL Image` 或者别的特定格式

- convert_inputs 用于把 LLM 传给 Tool 的内容解析成推理接口需要的格式，例如：
  - VisualChatGPT 使用 image_path 来传递图片
  - Transformer Agents 使用 PIL Image 来传递图片
  - 这个例子中的 Tool 输入是 image_path 格式，输出也是 image_path

因此，我们需要在 convert_inputs 提供不同 LLM 格式的解析：

```Python
def convert_inputs(self, inputs, **kwargs):
    if self.input_style == 'image_path':  # visual chatgpt style
        return inputs
    elif self.input_style == 'pil image':  # transformer agent style
        temp_image_path = get_new_file_path(
            'image/temp.jpg', func_name='temp')
        inputs.save(temp_image_path)
        return temp_image_path
    else:
        raise NotImplementedError
```

同理，在 Tool 完成推理后也需要 convert_outputs 转为 LLM 需要的格式：

```Python
def convert_outputs(self, outputs, **kwargs):
    if self.output_style == 'image_path':  # visual chatgpt style
        return outputs
    elif self.output_style == 'pil image':  # transformer agent style
        from PIL import Image
        outputs = Image.open(outputs)
        return outputs
    else:
        raise NotImplementedError
```

- 默认情况下 convert_inputs 和 convert_outputs 都会直接 return inputs 和 return outputs。
- 如果你定义了一个新的 `input_style` 或 `output_style`，你需要到 `tools/base_tool.py` 下更新对应的 `generate_xxx_description()`，用来为该类型自动生成格式描述。

### 4. 实现 apply

最重要的是实现 apply ，apply 是整个工具推理的核心代码。

对于ImageCaption工具而言，输入输出都是文本，所以实现比较简单，大家注意 scope 的切换就好

```python
def apply(self, inputs, **kwargs):
    if self.remote:
        raise NotImplementedError
    else:
        with Registry('scope').switch_scope_and_registry('mmpretrain'):
            outputs = self._inferencer(inputs)[0]['pred_caption']
    return outputs
```

对于检测工具 GLIP 而言，输入有两个：image_path 和 text，在convert_inputs已经完成了解析

```Python
def apply(self, inputs, **kwargs):
    image_path, text = inputs
    if self.remote:
        ...
    else:
        with Registry('scope').switch_scope_and_registry('mmdet'):
                results = self._inferencer(imgs=image_path, text_prompt=text)
                output_path = get_new_file_path(
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
