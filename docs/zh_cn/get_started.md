# 安装

根据您的情况选择安装方法：

## 安装完整工具包

安装完整工具包，你可以直接使用几乎所有工具（除了 ImageBind 和 SAM 等工具需要额外的依赖）。

1. 设置您的 torch 环境（如果已经完成则跳过此步骤）

```bash
conda create -n agentlego python=3.10
```

并按照[官方指南](https://pytorch.org/get-started/locally/#start-locally)安装 PyTorch 包（包括 torch,
torchvision 和 torchaudio）。

2. 安装 AgentLego 和一些常见的依赖项。

```bash
pip install agentlego[optional] openmim

# 用于图像理解工具。
pip install mmpretrain mmdet mmpose easyocr

# 用于图像生成工具。
pip install transformers diffusers
```

3. 某些工具需要额外的依赖项，在使用之前请查看 `Tool APIs` 中的 **Set up** 部分。

## 仅安装最简依赖

仅安装最简依赖，您可以使用类似于 GoogleSearch、Translation 和您自己的自定义工具的部分工具。此外，如果您从远程工具服务器调用工具，则客户端只需要简单的依赖项。

```bash
pip install agentlego
```

# 快速开始

## 直接使用工具

您可以在 AgentLego 中直接调用所有工具。

```Python
from agentlego import list_tools, load_tool

# 列出 AgentLego 中的所有工具
print(list_tools())

# 加载要使用的工具
calculator_tool = load_tool('Calculator')
print(calculator_tool.description)

# 直接调用工具
print(calculator_tool('cos(pi / 6)'))

# 图像或音频输入支持多种格式
from PIL import Image

image_caption_tool = load_tool('ImageDescription', device='cuda')
img_path = './examples/demo.png'
img_pil = Image.open(img_path)
print(image_caption_tool(img_path))
print(image_caption_tool(img_pil))
```

## 集成到智能体框架

### Lagent

[Lagent](https://github.com/InternLM/lagent) 是一个轻量级的开源框架，允许用户高效地构建基于大型语言模型（LLM）的智能体。

以下是一个示例脚本，将 agentlego 工具集成到 Lagent 中：

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.tools import Calculator

# 加载您想要使用的工具
tools = [Calculator().to_lagent()]

# 构建 Lagent 智能体
model = GPTAPI(temperature=0.)
agent = ReAct(llm=model, action_executor=ActionExecutor(tools))

user_input = '如果三角形的边长分别为 3cm、4cm 和 5cm，请告诉我三角形的面积。'
ret = agent.chat(user_input)

# 打印所有中间步骤结果
for step in ret.inner_steps[1:]:
   print('------')
   print(step['content'])
```

### LangChain

[LangChain](https://python.langchain.com/docs/get_started/introduction) 是一个用于创建利用语言模型的应用程序的框架。它提供了上下文感知和推理能力，为与语言模型一起使用和针对特定任务的预定义链提供了组件，使用户可以轻松地开始和自定义应用程序。

以下是一个示例脚本，将 agentlego 工具集成到 LangChain 中：

```python
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from agentlego.tools import Calculator

# 加载要使用的工具。
tools = [Calculator().to_langchain()]

# 构建 LangChain 智能体链
llm = ChatOpenAI(temperature=0.)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
agent = create_structured_chat_agent(llm, tools, prompt=hub.pull("hwchase17/structured-chat-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

user_input = '如果三角形的边长分别为3cm、4cm和5cm，请用工具计算三角形的面积。'
agent_executor.invoke(dict(input=user_input))
```

### Transformers Agent

[HuggingFace Transformers agent](https://huggingface.co/docs/transformers/transformers_agents) 是一个可扩展的自然语言 API，它理解用户输入并挑选工具进行操作。它允许轻松地集成由社区开发的其他工具。

以下是一个示例脚本，将 agentlego 工具集成到 Transformers agent 中：

```python
from transformers import HfAgent
from agentlego.tools import Calculator

# 加载要使用的工具
tools = [Calculator().to_transformers_agent()]

# 构建 Transformers Agent
prompt = open('examples/hf_agent/hf_demo_prompts.txt', 'r').read()
agent = HfAgent(
   'https://api-inference.huggingface.co/models/bigcode/starcoder',
   chat_prompt_template=prompt,
   additional_tools=tools,
)

user_input = '如果三角形的边长分别为3厘米、4厘米和5厘米，请告诉我三角形的面积。'
agent.chat(user_input)
```
