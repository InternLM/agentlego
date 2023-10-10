# HumanBodyPose

## Default Tool Meta

- **name**: Human Body Pose Detection On Image
- **description**: This tool can estimate the pose or keypoints of human in an image and draw the human pose image
- **inputs**: image
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/human.jpg
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('HumanBodyPose', device='cuda')

# apply tool
image = tool('human.jpg')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['HumanBodyPose'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'human.jpg'
ret = agent.chat(f'Extract pose of the human in the image {img_path}')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U openmim
pip install git+https://github.com/jin-s13/xtcocoapi
mim install -U mmpose
```

## Reference

This tool uses a **RTM Pose** model in default settings. See the following paper for details.

```bibtex
@misc{jiang2023rtmpose,
      title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
      author={Tao Jiang and Peng Lu and Li Zhang and Ningsheng Ma and Rui Han and Chengqi Lyu and Yining Li and Kai Chen},
      year={2023},
      eprint={2303.07399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# PoseToImage

## Default Tool Meta

- **name**: Generate Image Condition On Pose Image
- **description**: This tool can generate an image from a human pose image and a text. The text should be a series of English keywords separated by comma.
- **inputs**: image, text
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/pose_demo.jpg
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool
from PIL import Image

# load tool
tool = load_tool('PoseToImage', device='cuda')

# apply tool
image = tool('pose_demo.jpg', 'A pretty dancing girl.')
print(image)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['PoseToImage'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'pose_demo.jpg'
ret = agent.chat(f'According to the pose image `{img_path}`, draw a pretty dancing girl.')
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

# HumanFaceLandmark

## Default Tool Meta

- **name**: Human Face Landmark On Image
- **description**: This tool can estimate the landmark or keypoints of human faces in an image and draw the image with landmarks.
- **inputs**: image
- **outputs**: image

## Examples

**Download the demo resource**

```bash
wget http://download.openmmlab.com/mmtools/face.png
```

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool
from PIL import Image

# load tool
tool = load_tool('HumanFaceLandmark', device='cuda')

# apply tool
face_landmark = tool('face.png')
print(face_landmark)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['HumanFaceLandmark'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
img_path = 'face.png'
ret = agent.chat(f'Draw the face landmark of the human in the image `{img_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U openmim
pip install git+https://github.com/jin-s13/xtcocoapi
mim install -U mmpose
```

## Reference

This tool uses a **RTM Pose** model in default settings. See the following paper for details.

```bibtex
@misc{jiang2023rtmpose,
      title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
      author={Tao Jiang and Peng Lu and Li Zhang and Ningsheng Ma and Rui Han and Chengqi Lyu and Yining Li and Kai Chen},
      year={2023},
      eprint={2303.07399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
