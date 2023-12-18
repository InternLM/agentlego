# OCR

## Examples

**Download the demo resource**

```bash
wget https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_kie.jpeg
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('OCR', device='cuda', lang='en', x_ths=3., line_group_tolerance=30)

# apply tool
res = tool('demo_kie.jpeg')
```
For bilingual Chinese and English OCR, `lang` may be `['en', 'ch_sim']`, [here](https://www.jaided.ai/easyocr/) is all supported language code name.

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('OCR', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Here is a receipt image `demo_kie.jpeg`, please tell me the total cost.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install easyocr
```

## Reference

The default implementation of OCR tool uses [EasyOCR](https://github.com/JaidedAI/EasyOCR).
