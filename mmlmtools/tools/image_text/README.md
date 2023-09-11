# Image and Text

## Image Caption

This tool describes the content of the input image in natural language.

### API Refenrece

```python
class ImageCaption(BaseTool):
    '''Image captioning tool.

    Args:
        toolmeta (ToolMeta)
        parser (BaseParser)
        remote (bool)
        device (str)
    '''

    DEFAULT_TOOLMETA = dict(
        name='Get Photo Description',
        model={'model': 'blip-base_3rdparty_caption'},
        description='This is a useful tool when you want to know '
        'what is inside the image. It takes an {{{input:image}}} as the '
        'input, and returns a {{{output:text}}} representing the description '
        'of the image. ')

```

### Examples

#### Offline (w/o. Agent)

```python
import cv2
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('ImageCaption')

# apply tool
img = cv2.imread('my_image.png')
caption = tool(img)
```

#### HuggingFace Agent

```python
from mmlmtools.apis.agents.transformers_agent import load_tools_for_hfagent
from transformers import HfAgent
import cv2

# load tools and build huggingface agent
tools = load_tools_for_hfagent(tool_names=['ImageCaption'])
agent = HfAgent('https://api-inference.huggingface.co/models/bigcode/starcoder', additional_tools=tools)

# agent running with the tool
agent.run('What is in the image?', image=cv2.imread('my_image.png'))
```

### Set Up

Install dependencies to use the tool:

```bash
pip install -U openmim
mim install -U mmpretrain
```

### Attribution

This tool uses a *BLIP* model in default settings. See the following paper for details.

```bibtex
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
```

## TextToImage

This tool generates image from text descriptions.

### Example

#### Offline (w/o. Agent)

WIP.

#### HuggingFace Agent

WIP.

### Set Up

Install dependencies to use the tool:

```bash
pip install -U openmim
mim install -U mmagic
```

### Attribution

WIP.
