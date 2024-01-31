# ImageToCanny

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageToCanny')

# apply tool
canny = tool('examples/demo.png')
print(canny)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ImageToCanny').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Please do edge detection on the image `{img_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

# CannyTextToImage

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/agentlego/canny.png
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('CannyTextToImage', device='cuda')

# apply tool
image = tool('canny.png')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('CannyTextToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
canny = 'canny.png'
ret = agent.chat(f'According to the canny edge `{canny}`, draw a cartoon style image.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
```

## Reference

This tool uses a **Control Net** model in default settings. See the following paper for details.

```bibtex
@misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models},
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
