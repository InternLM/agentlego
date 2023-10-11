# ImageCaption

## Default Tool Meta

- **name**: Image Description
- **description**: A useful tool that returns a brief description of the input image.
- **inputs**: image
- **outputs**: text

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageCaption', device='cuda')

# apply tool
caption = tool('examples/demo.png')
print(caption)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['ImageCaption'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Describe the image `{img_path}`.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

**With HuggingFace Agent**

```python
from transformers import HfAgent
from agentlego.apis.agents import load_tools_for_hfagent
from PIL import Image

# load tools and build huggingface agent
tools = load_tools_for_hfagent(tools=['ImageCaption'], device='cuda')
agent = HfAgent('https://api-inference.huggingface.co/models/bigcode/starcoder', additional_tools=tools)

# agent running with the tool (For demo, we directly specify the tool name here.)
tool_name = tools[0].name
caption = agent.run(f'Use the tool `{tool_name}` to describe the image.', image=Image.open('examples/demo.png'))
print(caption)
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U openmim
mim install -U mmpretrain
```

## Reference

This tool uses a **BLIP** model in default settings. See the following paper for details.

```bibtex
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
```

# TextToImage

## Default Tool Meta

- **name**: Generate Image From Text
- **description**: This tool can generate an image according to the input text. The input text should be a series of keywords separated by comma, and all keywords must be in English.
- **inputs**: text
- **outputs**: image

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('TextToImage', device='cuda')

# apply tool
image = tool('cute cat')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['TextToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'Please generate an image of cute cat.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

**With HuggingFace Agent**

```python
from transformers import HfAgent
from agentlego.apis.agents import load_tools_for_hfagent

# load tools and build huggingface agent
tools = load_tools_for_hfagent(tools=['TextToImage'], device='cuda')
agent = HfAgent('https://api-inference.huggingface.co/models/bigcode/starcoder', additional_
tools=tools)

# agent running with the tool (For demo, we directly specify the tool name here.)
tool_name = tools[0].name
image = agent.run(f'Use the tool `{tool_name}` to generate an image of cat.')
print(image)
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U openmim
mim install -U mmagic
```

## Reference

This tool uses a **Stable Diffusion** model in default settings. See the following paper for details.

```bibtex
@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
```
