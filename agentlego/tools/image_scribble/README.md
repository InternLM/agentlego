# ImageToScribble

```{eval-rst}
.. autoclass:: agentlego.tools.ImageToScribble
    :noindex:
```

## Default Tool Meta

- **name**: Generate Scribble Conditioned On Image
- **description**: This tool can generate a sketch scribble of an image.
- **inputs**: image
- **outputs**: image

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageToScribble', device='cuda')

# apply tool
scribble = tool('examples/demo.png')
print(scribble)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ImageToScribble'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Please draw the scribble sketch of the image `{img_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U controlnet_aux
```

## Reference

The tool use [`controlnet_aux`](https://github.com/patrickvonplaten/controlnet_aux) to generate an auxiliary
scribble sketch that can be used in the ControlNet.

# ScribbleTextToImage

```{eval-rst}
.. autoclass:: agentlego.tools.ScribbleTextToImage
    :noindex:
```

## Default Tool Meta

- **name**: Generate Image Condition On Scribble Image
- **description**: This tool can generate an image from a sketch scribble image and a description.
- **inputs**: image, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/scribble.png
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ScribbleTextToImage', device='cuda')

# apply tool
image = tool('scribble.png', 'a pair of cartoon style pets')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ScribbleTextToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'scribble.png'
ret = agent.chat(f'According to the scribble sketch `{img_path}`, draw a pair of cartoon style cat and dog.')
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
