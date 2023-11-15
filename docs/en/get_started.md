# Installation

You can choose the installation method according to your condition:

## With deep-learning model support

1. Set up your torch environment (skip the step if you have already finished)

```bash
conda create -n agentlego python=3.10
```

And install PyTorch packages according to the [official guide](https://pytorch.org/get-started/locally/#start-locally).

2. Install AgentLego and some common dependencies.

```bash
pip install agentlego[optional] openmim

# For image understanding tools.
pip install mmpretrain mmdet mmpose easyocr

# For image generation tools.
pip install transformers diffusers mmagic
```

3. Some tools requires extra dependencies, check the **Set up** section in `Tool APIs` before you want to use.

## With simple dependencies

In this condition, you can use partial tools like GoogleSearch, Translation, and your custom tools. Moreover,
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

image_caption_tool = load_tool('ImageCaption', device='cuda')
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
from agentlego.apis import load_tool
from lagent import ReAct, GPTAPI, ActionExecutor

# Load the tools you want to use.
tool = load_tool('Calculator').to_lagent()

# Build Lagent Agent
model = GPTAPI(temperature=0.)
agent = ReAct(llm=model, action_executor=ActionExecutor([tool]))

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
from agentlego.apis import load_tool
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Load the tools you want to use.
tool = load_tool('Calculator').to_langchain()

# Build LangChain Agent
model = ChatOpenAI(temperature=0.)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
agent = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    llm=model,
    tools=[tool],
    memory=memory,
    verbose=True,
)

user_input = 'If the side lengths of a triangle are 3cm, 4cm and 5cm, please tell me the area of the triangle.'
agent.run(input=user_input)
```

### Transformers Agent

[HuggingFace Transformers agent](https://huggingface.co/docs/transformers/transformers_agents) is an
extensible natural language API which interprets language and uses curated tools for its operation. It allows
easy incorporation of additional community-developed tools.

Here is an example script to integrate agentlego tools to Transformers agent:

```python
from agentlego.apis import load_tool
from transformers import HfAgent

# Load the tools you want to use.
tool = load_tool('Calculator').to_transformers_agent()

# Build HuggingFace Transformers Agent
prompt = open('examples/hf_agent/hf_demo_prompts.txt', 'r').read()
agent = HfAgent(
    'https://api-inference.huggingface.co/models/bigcode/starcoder',
    chat_prompt_template=prompt,
    additional_tools=[tool],
)

user_input = 'If the side lengths of a triangle are 3cm, 4cm and 5cm, please tell me the area of the triangle.'
agent.chat(user_input)
```

# Tool Server

AgentLego provides a suit of tool server utilities to help you deploy tools on a server and use it like local
tools on clients.

## Start a server

We provide a script `server.py` to start a tool server. You can specify the tool names you want to use.

```bash
python server.py Calculator ImageCaption TextToImage
```

And then, the server will setup all tools and start.

```bash
INFO:     Started server process [1741344]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:16180 (Press CTRL+C to quit)
```

## Use tools in client

In the client, you can create a remote tool from the url of the tool server.

```python
from agentlego.tools.remote import RemoteTool

# Create all remote tools from a tool server root url.
tools = RemoteTool.from_server('http://127.0.0.1:16180')
for tool in tools:
    print(tool.name, tool.url)

# Create single remote tool from a tool server endpoint.
# All endpoint can be found in the docs of the tool server, like http://127.0.0.1:16180/docs
tool = RemoteTool('http://127.0.0.1:16180/ImageDescription')
print(tool.description)
```
