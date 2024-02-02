# ImageDescription

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageDescription', device='cuda')

# apply tool
caption = tool('examples/demo.png')
print(caption)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ImageDescription', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'Describe the image `{img_path}`.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

**With Transformers Agent**

```python
from transformers import HfAgent
from agentlego.apis import load_tool
from PIL import Image

# load tools and build transformers agent
tool = load_tool('ImageDescription', device='cuda').to_transformers_agent()
agent = HfAgent('https://api-inference.huggingface.co/models/bigcode/starcoder', additional_tools=[tool])

# agent running with the tool (For demo, we directly specify the tool name here.)
caption = agent.run(f'Use the tool `{tool.name}` to describe the image.', image=Image.open('examples/demo.png'))
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
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('TextToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please generate an image of cute cat.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

**With Transformers Agent**

```python
from transformers import HfAgent
from agentlego.apis import load_tool

# load tools and build transformers agent
tool = load_tool('TextToImage', device='cuda').to_transformers_agent()
agent = HfAgent('https://api-inference.huggingface.co/models/bigcode/starcoder', additional_tools=[tool])

# agent running with the tool (For demo, we directly specify the tool name here.)
image = agent.run(f'Use the tool `{tool.name}` to generate an image of cat.')
print(image)
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
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
