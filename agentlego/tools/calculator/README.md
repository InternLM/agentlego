# Calculator

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
from agentlego.apis import load_tool
from lagent import ReAct, GPTAPI, ActionExecutor

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('Calculator').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'pi and 3.2, which is greater?')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```
