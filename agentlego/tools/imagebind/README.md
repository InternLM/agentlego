# AudioToImage

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/agentlego/cat.wav
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioToImage', device='cuda')

# apply tool
image = tool('./cat.wav')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('AudioToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please generate an image according to the audio at `cat.wav`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers pytorchvideo
```

## Reference

This tool uses a **ImageBind** model. See the following paper for details.

```bibtex
@misc{girdhar2023imagebind,
      title={ImageBind: One Embedding Space To Bind Them All},
      author={Rohit Girdhar and Alaaeldin El-Nouby and Zhuang Liu and Mannat Singh and Kalyan Vasudev Alwala and Armand Joulin and Ishan Misra},
      year={2023},
      eprint={2305.05665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# ThermalToImage

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/agentlego/thermal.jpg
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ThermalToImage', device='cuda')

# apply tool
image = tool('thermal.jpg')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ThermalToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please generate an image according to the thermal image at `thermal.jpg`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers
```

## Reference

This tool uses a **ImageBind** model. See the following paper for details.

```bibtex
@misc{girdhar2023imagebind,
      title={ImageBind: One Embedding Space To Bind Them All},
      author={Rohit Girdhar and Alaaeldin El-Nouby and Zhuang Liu and Mannat Singh and Kalyan Vasudev Alwala and Armand Joulin and Ishan Misra},
      year={2023},
      eprint={2305.05665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# AudioImageToImage

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/agentlego/dog.jpg
wget http://download.openmmlab.com/agentlego/cat.wav
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioImageToImage', device='cuda')

# apply tool
image = tool('dog.jpg', 'cat.wav')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('AudioImageToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please generate an image according to the audio `cat.wav` and image `dog.jpg`.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers pytorchvideo
```

## Reference

This tool uses a **ImageBind** model. See the following paper for details.

```bibtex
@misc{girdhar2023imagebind,
      title={ImageBind: One Embedding Space To Bind Them All},
      author={Rohit Girdhar and Alaaeldin El-Nouby and Zhuang Liu and Mannat Singh and Kalyan Vasudev Alwala and Armand Joulin and Ishan Misra},
      year={2023},
      eprint={2305.05665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# AudioTextToImage

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/agentlego/cat.wav
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('AudioTextToImage', device='cuda')

# apply tool
image = tool('cat.wav', 'flying in the sky')
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('AudioTextToImage', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
ret = agent.chat(f'Please generate an image according to the audio `cat.wav`, and it should fly in the sky.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install timm ftfy iopath diffusers pytorchvideo
```

## Reference

This tool uses a **ImageBind** model. See the following paper for details.

```bibtex
@misc{girdhar2023imagebind,
      title={ImageBind: One Embedding Space To Bind Them All},
      author={Rohit Girdhar and Alaaeldin El-Nouby and Zhuang Liu and Mannat Singh and Kalyan Vasudev Alwala and Armand Joulin and Ishan Misra},
      year={2023},
      eprint={2305.05665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
