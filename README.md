<div align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26739999/289025203-f05733ff-6bbb-46f0-92aa-8827c59df79c.png" width="450"/>
</div>

<div align="center">

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/mzr1996/AgentLego)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://agentlego.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/agentlego)](https://pypi.org/project/agentlego)
[![license](https://img.shields.io/github/license/InternLM/agentlego.svg)](https://github.com/InternLM/agentlego/tree/main/LICENSE)

English | [简体中文](./README_zh-CN.md)

</div>

- [Introduction](#introduction)
- [Quick Starts](#quick-starts)
  - [Installation](#installation)
  - [Use tools directly](#use-tools-directly)
  - [Integrated into agent frameworks](#integrated-into-agent-frameworks)
- [Supported Tools](#supported-tools)
- [Licence](#licence)

## Introduction

<span style="color:blue"> *AgentLego* </span> is an open-source library of versatile tool APIs to extend and enhance large language model (LLM) based agents, with the following highlight features:

- **Rich set of tools for multimodal extensions of LLM agents** including visual perception, image generation and editing, speech processing and visual-language reasoning, etc.
- **Flexible tool interface** that allows users to easily extend custom tools with arbitrary types of arguments and outputs.
- **Easy integration with LLM-based agent frameworks** like [LangChain](https://github.com/langchain-ai/langchain), [Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents), [Lagent](https://github.com/InternLM/lagent).
- **Support tool serving and remote accessing**, which is especially useful for tools with heavy ML models (e.g. ViT) or special environment requirements (e.g. GPU and CUDA).

https://github-production-user-asset-6210df.s3.amazonaws.com/26739999/289006700-2140015c-b5e0-4102-bc54-9a1b4e3db9ec.mp4

# Quick Starts

## Installation

**Install the AgentLego package**

```shell
pip install agentlego
```

**Install tool-specific dependencies**

Some tools requires extra packages, please check the readme file of the tool, and confirm all requirements are
satisfied.

For example, if we want to use the `ImageDescription` tool. We need to check the **Set up** section of
[readme](agentlego/tools/image_text/README.md#ImageDescription) and install the requirements.

```bash
pip install -U openmim
mim install -U mmpretrain
```

## Use tools directly

```Python
from agentlego import list_tools, load_tool

print(list_tools())  # list tools in AgentLego

image_caption_tool = load_tool('ImageDescription', device='cuda')
print(image_caption_tool.description)
image = './examples/demo.png'
caption = image_caption_tool(image)
```

## Integrated into agent frameworks

- [**Lagent**](examples/lagent_example.py)
- [**Transformers Agent**](examples/hf_agent/hf_agent_example.py)
- [**VisualChatGPT**](examples/visual_chatgpt/visual_chatgpt.py)

# Supported Tools

**General ability**

- [Calculator](agentlego/tools/calculator/README.md): Calculate by Python interpreter.
- [GoogleSearch](agentlego/tools/search/README.md): Search on Google.

**Speech related**

- [TextToSpeech](agentlego/tools/speech_text/README.md#TextToSpeech): Speak the input text into audio.
- [SpeechToText](agentlego/tools/speech_text/README.md#SpeechToText): Transcribe an audio into text.

**Image-processing related**

- [ImageDescription](agentlego/tools/image_text/README.md#ImageDescription): Describe the input image.
- [OCR](agentlego/tools/ocr/README.md#OCR): Recognize the text from a photo.
- [VQA](agentlego/tools/vqa/README.md#VQA): Answer the question according to the image.
- [HumanBodyPose](agentlego/tools/image_pose/README.md#HumanBodyPose): Estimate the pose or keypoints of human in an image.
- [HumanFaceLandmark](agentlego/tools/image_pose/README.md#HumanFaceLandmark): Estimate the landmark or keypoints of human faces in an image.
- [ImageToCanny](agentlego/tools/image_canny/README.md#ImageToCanny): Extract the edge image from an image.
- [ImageToDepth](agentlego/tools/image_depth/README.md#ImageToDepth): Generate the depth image of an image.
- [ImageToScribble](agentlego/tools/image_scribble/README.md#ImageToScribble): Generate a sketch scribble of an image.
- [ObjectDetection](agentlego/tools/object_detection/README.md#ObjectDetection): Detect all objects in the image.
- [TextToBbox](agentlego/tools/object_detection/README.md#TextToBbox): Detect specific objects described by the given text in the image.
- Segment Anything series
  - [SegmentAnything](agentlego/tools/segmentation/README.md#SegmentAnything): Segment all items in the image.
  - [SegmentObject](agentlego/tools/segmentation/README.md#SegmentObject): Segment the certain objects in the image according to the given object name.

**AIGC related**

- [TextToImage](agentlego/tools/image_text/README.md#TextToImage): Generate an image from the input text.
- [ImageExpansion](agentlego/tools/image_editing/README.md#ImageExpansion): Expand the peripheral area of an image based on its content.
- [ObjectRemove](agentlego/tools/image_editing/README.md#ObjectRemove): Remove the certain objects in the image.
- [ObjectReplace](agentlego/tools/image_editing/README.md#ObjectReplace): Replace the certain objects in the image.
- [ImageStylization](agentlego/tools/image_editing/README.md#ImageStylization): Modify an image according to the instructions.
- ControlNet series
  - [CannyTextToImage](agentlego/tools/image_canny/README.md#CannyTextToImage): Generate an image from a canny edge image and a description.
  - [DepthTextToImage](agentlego/tools/image_depth/README.md#DepthTextToImage): Generate an image from a depth image and a description.
  - [PoseToImage](agentlego/tools/image_pose/README.md#PoseToImage): Generate an image from a human pose image and a description.
  - [ScribbleTextToImage](agentlego/tools/image_scribble/README.md#ScribbleTextToImage): Generate an image from a sketch scribble image and a description.
- ImageBind series
  - [AudioToImage](agentlego/tools/imagebind/README.md#AudioToImage): Generate an image according to audio.
  - [ThermalToImage](agentlego/tools/imagebind/README.md#ThermalToImage): Generate an image according a thermal image.
  - [AudioImageToImage](agentlego/tools/imagebind/README.md#AudioImageToImage): Generate am image according to a audio and image.
  - [AudioTextToImage](agentlego/tools/imagebind/README.md#AudioTextToImage): Generate an image from a audio and text prompt.

# Licence

This project is released under the [Apache 2.0 license](LICENSE). Users should also ensure compliance with the licenses governing the models used in this project.
