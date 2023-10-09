# ImageBind

# AudioToImage

```{eval-rst}
.. autoclass:: agentlego.tools.AudioToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Audio
- **description**: This tool can generate an image according to the input audio
- **inputs**: audio
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['AudioToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'TODO')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers
```

## Reference

TODO

# ThermalToImage

```{eval-rst}
.. autoclass:: agentlego.tools.ThermalToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Thermal Image
- **description**: This tool can generate an image according to the input thermal image.
- **inputs**: image
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ThermalToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ThermalToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'TODO')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers
```

## Reference

TODO

# AudioImageToImage

```{eval-rst}
.. autoclass:: agentlego.tools.AudioImageToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Image and Audio
- **description**: This tool can generate an image according to the input reference image and the input audio.
- **inputs**: image, audio
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioImageToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['AudioImageToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'TODO')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers
```

## Reference

TODO

# AudioTextToImage

```{eval-rst}
.. autoclass:: agentlego.tools.AudioTextToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Audio and Text
- **description**: This tool can generate an image according to the input audio and the input description.
- **inputs**: audio, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioTextToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['AudioTextToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'TODO')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers
```

## Reference

TODO
