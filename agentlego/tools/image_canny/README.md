# ImageToCanny

## Default Tool Meta

- **name**: Edge Detection On Image
- **description**: This tool can extract the edge image from an image.
- **inputs**: image
- **outputs**: image

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
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ImageToCanny'])
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Please do edge detection on the image `{img_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

# CannyTextToImage

## Default Tool Meta

- **name**: Generate Image Condition On Canny Image
- **description**: This tool can generate an image from a canny edge image and a description.
- **inputs**: image, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/canny.png
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
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['CannyTextToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

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
pip install -U openmim
mim install -U mmagic
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
