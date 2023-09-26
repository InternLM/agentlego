# ObjectDetection

```{eval-rst}
.. autoclass:: agentlego.tools.ObjectDetection
    :noindex:
```

## Default Tool Meta

- **name**: Detect All Objects
- **description**: A useful tool when you only want to detect the picture or detect all objects in the picture. like: detect all objects.
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
tool = load_tool('ObjectDetection', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ObjectDetection'], device='cuda')
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

# TextToBbox

```{eval-rst}
.. autoclass:: agentlego.tools.TextToBbox
    :noindex:
```

## Default Tool Meta

- **name**: Detect the Given Object
- **description**: A useful tool when you only want to show the location of given objects, or detect or find out given objects in the picture. like: locate persons in the picture
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
tool = load_tool('TextToBbox', device='cuda')

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['TextToBbox'], device='cuda')
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
