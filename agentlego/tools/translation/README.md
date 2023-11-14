# Translation

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('Translation')

# apply tool
res = tool('This is an example sentence.', 'auto', 'zh-CN')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('Translation').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please translate the below text to Chinese: Nothing is true, everything is permitted.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```
