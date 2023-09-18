# VisualQuestionAnswering

```{eval-rst}
.. autoclass:: mmlmtools.tools.VisualQuestionAnswering
    :noindex:
```

## Default Tool Meta

- **name**: Visual Question Answering
- **description**: This tool can answer the question according to the image.
- **inputs**: image, text
- **outputs**: text

## Examples

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('VisualQuestionAnswering', device='cuda')

# apply tool
answer = tool('examples/demo.png','What is the color of the cat?')
print(answer)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['VisualQuestionAnswering'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'According to the image `{img_path}`, what is the color of the cat?')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using this tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U openmim
mim install -U mmpretrain
```

## Reference

This tool uses a **OFA** model in default settings. See the following paper for details.

```bibtex
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```