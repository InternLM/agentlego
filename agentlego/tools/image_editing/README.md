# ImageExpansion

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageExpansion', device='cuda')

# apply tool
image = tool('examples/demo.png', '1.25')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ImageExpansion', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'According to the image `{img_path}`, expand its size to 1.25 times')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using this tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
pip install -U openmim
mim install -U mmpretrain
```

## Reference

This tool uses **BLIP** and **Stable Diffusion** in default settings. See the following papers for details.

```bibtex
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
```

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

# ObjectRemove

## Example

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectRemove', device='cuda')

# apply tool
image = tool('examples/demo.png','dog')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ObjectRemove', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'According to the image `{img_path}`, remove the dog in the image.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using this tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
pip install -U segment_anything
pip install -U openmim
mim install -U mmdet
```

## Reference

This tool uses **SAM**, **Stable Diffusion** and **GLIP** in default settings. See the following papers for details.

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

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

```bibtex
@article{zhang2022glipv2,
  title={GLIPv2: Unifying Localization and Vision-Language Understanding},
  author={Zhang, Haotian* and Zhang, Pengchuan* and Hu, Xiaowei and Chen, Yen-Chun and Li, Liunian Harold and Dai, Xiyang and Wang, Lijuan and Yuan, Lu and Hwang, Jenq-Neng and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2206.05836},
  year={2022}
}
```

# ObjectReplace

## Example

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectReplace', device='cuda')

# apply tool
image = tool('examples/demo.png', 'cat', 'a white dog')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ObjectReplace', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'According to the image `{img_path}`, replace the cat with a white dog in the image.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using this tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
pip install -U segment_anything
pip install -U openmim
mim install -U mmdet
```

## Reference

This tool uses **SAM**, **Stable Diffusion** and **GLIP** in default settings. See the following papers for details.

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

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

```bibtex
@article{zhang2022glipv2,
  title={GLIPv2: Unifying Localization and Vision-Language Understanding},
  author={Zhang, Haotian* and Zhang, Pengchuan* and Hu, Xiaowei and Chen, Yen-Chun and Li, Liunian Harold and Dai, Xiyang and Wang, Lijuan and Yuan, Lu and Hwang, Jenq-Neng and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2206.05836},
  year={2022}
}
```

# ImageStylization

## Examples

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('ImageStylization', device='cuda')

# apply tool
image = tool('examples/demo.png', 'turn the cat into a cartoon cat')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis import load_tool

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tool = load_tool('ImageStylization', device='cuda').to_lagent()
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor([tool]))

# agent running with the tool.
img_path = 'examples/demo.png'
ret = agent.chat(f'According to the image `{img_path}`, turn the cat into a cartoon cat.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using this tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U diffusers
```

## Reference

This tool uses a **instruct-pix2pix** model in default settings. See the following paper for details.

```bibtex
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```
