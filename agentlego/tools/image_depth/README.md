# ImageToDepth

```{eval-rst}
.. autoclass:: agentlego.tools.ImageToDepth
    :noindex:
```

## Default Tool Meta

- **name**: Generate Depth Image On Image
- **description**: This tool can generate the depth image of an image.
- **inputs**: image
- **outputs**: image

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageToDepth', device='cuda')

# apply tool
depth = tool('examples/demo.png')
print(depth)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ImageToDepth'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Please estimate the depth of the image `{img_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U transformers
```

## Reference

This tool uses a **DPT** model in default settings. See the following paper for details.

```bibtex
@article{DBLP:journals/corr/abs-2103-13413,
  author    = {Ren{\'{e}} Ranftl and
               Alexey Bochkovskiy and
               Vladlen Koltun},
  title     = {Vision Transformers for Dense Prediction},
  journal   = {CoRR},
  volume    = {abs/2103.13413},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.13413},
  eprinttype = {arXiv},
  eprint    = {2103.13413},
  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-13413.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# DepthTextToImage

```{eval-rst}
.. autoclass:: agentlego.tools.DepthTextToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image Condition On Depth Image
- **description**: This tool can generate an image from a depth image and a text. The text should be a series of English keywords separated by comma.
- **inputs**: image, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/depth.png
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('DepthTextToImage', device='cuda')

# apply tool
image = tool('depth.png', 'a pair of cartoon style pets')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['DepthTextToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'depth.png'
ret = agent.chat(f'According to the depth image `{img_path}`, draw a cartoon style image.')
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
