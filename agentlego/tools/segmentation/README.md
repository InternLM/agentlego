# SegmentAnything

## Default Tool Meta

- **name**: Segment Anything
- **description**: This tool can segment all items in the image and return a segmentation result image
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
tool = load_tool('SegmentAnything', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

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
pip install segment_anything
```

## Reference

TODO

# SegmentObject

## Default Tool Meta

- **name**: Segment specified object
- **description**: This tool can segment the specified kind of objects in the input image, and return the segmentation result image.
- **inputs**: image, text
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
tool = load_tool('SegmentObject', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['SegmentObject'], device='cuda')
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
pip install segment_anything
```

## Reference

TODO

# SemanticSegmentation

## Default Tool Meta

- **name**: Semantic Segment on urban scene
- **description**: This tool can segment all items in the input image and return a segmentation result image. It focus on urban scene images.
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
tool = load_tool('SemanticSegmentation', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['SemanticSegmentation'], device='cuda')
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
pip install mmsegmentation
```

## Reference

TODO

