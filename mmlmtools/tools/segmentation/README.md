# SegmentAnything

```{eval-rst}
.. autoclass:: mmlmtools.tools.SegmentAnything
    :noindex:
```

## Default Tool Meta

- **name**: Segment Anything On Image
- **description**: This is a useful tool when you want to segment anything in the image, like: segment anything from this image.
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
tool = load_tool('SegmentAnything', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['SegmentAnything'], device='cuda')
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

# SegmentClicked

```{eval-rst}
.. autoclass:: mmlmtools.tools.SegmentClicked
    :noindex:
```

## Default Tool Meta

- **name**: Segment The Clicked Region In The Image
- **description**: This is a useful tool when you want to segment the masked region or block in the image, like: segment the masked region in this image.
- **inputs**: image, mask
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
tool = load_tool('SegmentClicked', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['SegmentClicked'], device='cuda')
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

# ObjectSegmenting

```{eval-rst}
.. autoclass:: mmlmtools.tools.ObjectSegmenting
    :noindex:
```

## Default Tool Meta

- **name**: Segment The Given Object In The Image
- **description**: This is a useful tool when you want to segment the certain objects in the image according to the given object name, like: segment the cat in this image, or can you segment an object for me.
- **inputs**: image, text
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
tool = load_tool('ObjectSegmenting', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ObjectSegmenting'], device='cuda')
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
