# Translation

## Default Tool Meta

- **name**: Text translation
- **description**: This tool can translate a text from source language to the target language. The source_lang and target_lang can be one of 'auto' (Detect source language), 'zh-CN' (Chinese), 'en' (English), 'fr' (French), 'de' (German), 'el' (Greek), 'it' (Italian), 'ja' (Japanese), 'ko' (Korean), 'la' (Latin), 'pl' (Polish), 'ru' (Russian), 'es' (Spanish), 'th' (Thai), 'tr' (Turkish).
- **inputs**: text, text, text
- **outputs**: text

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
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['Translation'])
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'Please translate the below text to Chinese: Nothing is true, everything is permitted.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```
