# ImageBind

# AudioToImage

```{eval-rst}
.. autoclass:: mmlmtools.tools.AudioToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Audio
- **description**: This is a useful tool when you want to  generate a real image from audio. like: generate a real image from audio, or generate a new image based on the given audio.
- **inputs**: audio
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('AudioToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

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
TODO
```

## Reference

TODO

# ThermalToImage

```{eval-rst}
.. autoclass:: mmlmtools.tools.ThermalToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Thermal Image
- **description**: This is a useful tool when you want to  generate a real image from a thermal image. like: generate a real image from thermal image, or generate a new image based on the given thermal image.
- **inputs**: image
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('ThermalToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

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
TODO
```

## Reference

TODO

# AudioImageToImage

```{eval-rst}
.. autoclass:: mmlmtools.tools.AudioImageToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Image and Audio
- **description**: This is a useful tool when you want to generate a real image from image and audio. like: generate a real image from image and audio, or generate a new image based on the given image and audio.
- **inputs**: image, audio
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('AudioImageToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

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
TODO
```

## Reference

TODO

# AudioTextToImage

```{eval-rst}
.. autoclass:: mmlmtools.tools.AudioTextToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image from Audio and Text
- **description**: This is a useful tool when you want to  generate a real image from audio and text prompt. like: generate a real image from audio with user's prompt, or generate a new image based on the given image audio with user's description.
- **inputs**: audio, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('AudioTextToImage', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

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
TODO
```

## Reference

TODO
