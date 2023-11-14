# GoogleSearch

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('GoogleSearch')

# apply tool
res = tool('Highest mountain in the earth')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('GoogleSearch').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'What is the highest mountain in the earth?')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have a [Serper](https://serper.dev/) API key, and set it in
environment variables.

```bash
export SERPER_API_KEY='Your API key'
```

## Reference

The Google Search API comes from [Serper](https://serper.dev/)
