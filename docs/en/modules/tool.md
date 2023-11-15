# Implement a custom tool

The AgentLego is extensible and you can add your custom tools easily and apply them to various agent system.

## Simple Example

First, all tools should inherit the `BaseTool` class. As an example, assume we want a simple clock tool.

```python
from agentlego.tools import BaseTool
from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta

class Clock(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name='Clock',
            description='A clock that return the current date and time.',
            inputs=[],
            outputs=['text'],
        )
        super().__init__(toolmeta=toolmeta, parser=DefaultParser)
```

To initialize the tool, you need to construct a `ToolMeta` to specify the name, description, input arguments
categories and the output categories. The available categories are `text`, `image` and `audio` by now.

Then, you need also to specify a default parser, it's used to handle the input & output type. And usually, you
can directly use `DefaultParser` as the default parser.

Now, you can override the `setup` and `apply` method. The `setup` method will run when the tool is called at
the first time, and it's usually used to lazy-load some heavy modules. And the `apply` method is the core
method to perform when the tool is called. In this example, we only need to override the `apply` method.

```python
class Clock(BaseTool):
    ...

    def apply(self):
        from datetime import datetime
        return datetime.now().strftime('%Y/%m/%d %H:%M')
```

We have already finished the tool, now you can instantiate it and use it in agent systems.

```python
# Create an instance
tool = Clock()

# Use it in langchain
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

agent = initialize_agent(
    agent='structured-chat-zero-shot-react-description',
    llm=ChatOpenAI(temperature=0.),
    tools=[tool.to_langchain()],
    verbose=True)
agent.invoke("What's the time?")

# Use it in lagent
from lagent import ReAct, GPTAPI, ActionExecutor
agent = ReAct(
    llm=GPTAPI(temperature=0.),
    action_executor=ActionExecutor([tool.to_lagent()])
)
ret = agent.chat("What's the time?")
print(ret.response)
```

## Multi-modality input & output

A core feature of AgentLego is support multi-modality tools, and at the same time, we need to handle
multi-modality inputs and outputs. Different agent system accepts different formats, usually, language-based
agent system cannot accept image and audio, and we need to convert the input and output data to string, like
file path. But some agent systems can handle raw data directly, like Transformers Agent, and some agent
systems require the raw data to display at the front-end.

Therefore, we use agent types as the input & output types of the tool, and use a `parser` to convert to the
destination format automatically.

Assume we want a tool that can create an audio caption with the destination language on the input image.

```python
from agentlego.tools import BaseTool
from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO, AudioIO

class AudioCaption(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name='AudioCaption',
            description='A tool that can create an audio caption on the input image with the specified language.',
            inputs=['image', 'text'],
            outputs=['audio'],
        )
        super().__init__(toolmeta=toolmeta, parser=DefaultParser)

    def apply(self, image: ImageIO, language: str):
        # Convert the agent type to the format we need in the tool.
        image = image.to_pil()

        # pseudo-code of the processing.
        caption = image_caption(image)
        caption = translate(caption, language)
        audio_tensor = text_reader(caption, language)

        # Return the wrapped agent type.
        return AudioIO(audio_tensor)
```

The agent type can convert to supported format directly, can you can also construct the agent type from the
supported format. And the default `DefaultParser` will convert the input to the agent type and convert the
output agent type to string format output. So that, you can call the tool directly and use file path as
arguments.

```python
>>> tool = AudioCaption()
>>> audio_path = tool('examples/demo.png', 'en-US')
>>> print(audio_path)
generated/audio/20231011-1730.wav
```
