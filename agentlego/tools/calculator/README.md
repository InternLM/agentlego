# Calculator

## Default Tool Meta

- **name**: Calculator
- **description**: A calculator tool. The input must be a single Python expression and you cannot import packages. You can use functions in the `math` package without import.
- **inputs**: text
- **outputs**: text

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('Calculator')

# apply tool
tool('e ** cos(pi)')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['Calculator'])
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'pi and 3.2, which is greater?')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```
