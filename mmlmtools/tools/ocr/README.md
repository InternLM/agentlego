# OCR

```{eval-rst}
.. autoclass:: mmlmtools.tools.OCR
    :noindex:
```

## Default Tool Meta

- **name**: Recognize the Optical Characters On Image
- **description**: This is a useful tool when you want to recognize the text from a photo.
- **inputs**: image
- **outputs**: text

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('OCR', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['OCR'], device='cuda')
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

# ImageMaskOCR

```{eval-rst}
.. autoclass:: mmlmtools.tools.ImageMaskOCR
    :noindex:
```

## Default Tool Meta

- **name**: Recognize The Optical Characters On Image With Mask
- **description**: This is a useful tool when you want to recognize the characters or words in the masked region of the image. like: recognize the characters or words in the masked region.
- **inputs**: image, mask
- **outputs**: text

## Examples

**Download the demo resource**

```bash
TODO
```

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('ImageMaskOCR', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ImageMaskOCR'], device='cuda')
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
