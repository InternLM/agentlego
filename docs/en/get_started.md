# Installation

You can choose the installation method according to your condition:

## With full toolkits

With full toolkits, you can use almost all tools except few tools (ImageBind and SAM related tools requires
extra requirements) directly.

1. Set up your torch environment (skip the step if you have already finished)

```bash
conda create -n agentlego python=3.10
```

And install PyTorch packages (torch, torchvision and torchaudio) according to the [official guide](https://pytorch.org/get-started/locally/#start-locally).

2. Install AgentLego and some common dependencies.

```bash
pip install agentlego[optional] openmim

# For image understanding tools.
mim install mmpretrain mmdet mmpose easyocr

# For image generation tools.
pip install transformers diffusers
```

3. Some tools requires extra dependencies, check the **Set up** section in `Tool APIs` before you want to use.

## With minimum dependencies

With minimum dependencies, you can use partial tools like GoogleSearch, Translation, and your custom tools. Moreover,
if you call tools from a tool server remotely, the client only need simple dependencies.

```bash
pip install agentlego
```

# Quick Starts

## Use tools directly

You can all the tools in AgentLego directly.

```Python
from agentlego import list_tools, load_tool

# List all tools in AgentLego
print(list_tools())

# Load the tool to use
calculator_tool = load_tool('Calculator')
print(calculator_tool.description)

# Call the tool directly
print(calculator_tool('cos(pi / 6)'))

# Image or Audio input supports multiple formats
from PIL import Image

image_caption_tool = load_tool('ImageDescription', device='cuda')
img_path = './examples/demo.png'
img_pil = Image.open(img_path)
print(image_caption_tool(img_path))
print(image_caption_tool(img_pil))
```

## Integrated into agent frameworks

### Lagent

[Lagent](https://github.com/InternLM/lagent) is a lightweight open-source framework that allows users to
efficiently build large language model(LLM) -based agents.

Here is an example script to integrate agentlego tools to Lagent:

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.tools import Calculator

# Load the tools you want to use.
tools = [Calculator().to_lagent()]

# Build Lagent Agent
model = GPTAPI(temperature=0.)
agent = ReAct(llm=model, action_executor=ActionExecutor(tools))

user_input = 'If the side lengths of a triangle are 3cm, 4cm and 5cm, please tell me the area of the triangle.'
ret = agent.chat(user_input)

# Print all intermediate steps result
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

### LangChain

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for creating
applications leveraged by language models. It provides context-aware and reasoning abilities, offering
components for working with language models and preassembled chains for specific tasks, making it easy for
users to start and customize applications.

Here is an example script to integrate agentlego tools to LangChain:

```python
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from agentlego.tools import Calculator

# Load the tools you want to use.
tools = [Calculator().to_langchain()]

# Build LangChain Agent
llm = ChatOpenAI(temperature=0.)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
agent = create_structured_chat_agent(llm, tools, prompt=hub.pull("hwchase17/structured-chat-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

user_input = 'If the side lengths of a triangle are 3cm, 4cm and 5cm, please tell me the area of the triangle.'
agent_executor.invoke(dict(input=user_input))
```

### Transformers Agent

[HuggingFace Transformers agent](https://huggingface.co/docs/transformers/transformers_agents) is an
extensible natural language API which interprets language and uses curated tools for its operation. It allows
easy incorporation of additional community-developed tools.

Here is an example script to integrate agentlego tools to Transformers agent:

```python
from transformers import HfAgent
from agentlego.tools import Calculator

# Load the tools you want to use.
tools = [Calculator().to_transformers_agent()]

# Build HuggingFace Transformers Agent
prompt = open('examples/hf_agent/hf_demo_prompts.txt', 'r').read()
agent = HfAgent(
    'https://api-inference.huggingface.co/models/bigcode/starcoder',
    chat_prompt_template=prompt,
    additional_tools=tools,
)

user_input = 'If the side lengths of a triangle are 3cm, 4cm and 5cm, please tell me the area of the triangle.'
agent.chat(user_input)
```
