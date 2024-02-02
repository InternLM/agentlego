<div align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26739999/289025203-f05733ff-6bbb-46f0-92aa-8827c59df79c.png" width="450"/>
</div>

<div align="center">

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/mzr1996/AgentLego)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://agentlego.readthedocs.io/zh-cn/latest/)
[![PyPI](https://img.shields.io/pypi/v/agentlego)](https://pypi.org/project/agentlego)
[![license](https://img.shields.io/github/license/InternLM/agentlego.svg)](https://github.com/InternLM/agentlego/tree/main/LICENSE)

[English](./README.md) | 简体中文

</div>

- [简介](#简介)
- [快速开始](#快速开始)
  - [安装环境](#安装环境)
  - [直接使用工具](#直接使用工具)
  - [集成至智能体框架](#集成至智能体框架)
- [工具列表](#工具列表)
- [开源许可证](#开源许可证)

# 简介

<span style="color:blue"> *Agent Lego* </span> 是一个开源的多功能工具 API 库，用于扩展和增强基于大型语言模型（LLM）的智能体（Agent），具有以下突出特点：

- **丰富的多模态扩展工具集**，包括视觉感知、图像生成和编辑、语音处理和视觉语言推理等。
- **灵活的工具接口**，允许用户轻松扩展具有任意类型参数和输出的自定义工具。
- **与基于LLM的代理程序框架轻松集成**，如 [LangChain](https://github.com/langchain-ai/langchain)、[Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents)、[Lagent](https://github.com/InternLM/lagent)。
- **支持部署工具服务和远程访问**，这对于需要大型机器学习模型（例如 ViT）或特殊环境（例如 GPU 和 CUDA）的工具特别有用。

https://github-production-user-asset-6210df.s3.amazonaws.com/26739999/289006700-2140015c-b5e0-4102-bc54-9a1b4e3db9ec.mp4

# 快速开始

## 安装环境

**安装 AgentLego 包**

```shell
pip install agentlego
```

**安装工具特定的依赖**

一些工具需要额外的软件包，请查看工具的自述文件，并确认所有要求都得到满足。

例如，如果我们想要使用`ImageDescription`工具。我们需要查看工具 [readme](agentlego/tools/image_text/README.md#ImageDescription) 的 **Set up** 小节并安装所需的软件。

```bash
pip install -U openmim
mim install -U mmpretrain
```

## 直接使用工具

```Python
from agentlego import list_tools, load_tool

print(list_tools())  # list tools in AgentLego

image_caption_tool = load_tool('ImageDescription', device='cuda')
print(image_caption_tool.description)
image = './examples/demo.png'
caption = image_caption_tool(image)
```

## 集成至智能体框架

- [**Lagent**](examples/lagent_example.py)
- [**Transformers Agent**](examples/hf_agent/hf_agent_example.py)
- [**VisualChatGPT**](examples/visual_chatgpt/visual_chatgpt.py)

# 工具列表

**通用能力**

- [Calculator](agentlego/tools/calculator/README.md): 使用 Python 解释器进行计算
- [GoogleSearch](agentlego/tools/search/README.md): 使用 Google 搜索

**语音相关**

- [TextToSpeech](agentlego/tools/speech_text/README.md#TextToSpeech): 将输入文本转换为音频。
- [SpeechToText](agentlego/tools/speech_text/README.md#SpeechToText): 将音频转录为文本。

**图像处理相关**

- [ImageDescription](agentlego/tools/image_text/README.md#ImageDescription): 描述输入图像。
- [OCR](agentlego/tools/ocr/README.md#OCR): 从照片中识别文本。
- [VQA](agentlego/tools/vqa/README.md#VQA): 根据图片回答问题。
- [HumanBodyPose](agentlego/tools/image_pose/README.md#HumanBodyPose): 估计图像中人体的姿态或关键点，并绘制人体姿态图像
- [HumanFaceLandmark](agentlego/tools/image_pose/README.md#HumanFaceLandmark): 识别图像中人脸的关键点，并绘制带有关键点的图像。
- [ImageToCanny](agentlego/tools/image_canny/README.md#ImageToCanny): 从图像中提取边缘图像。
- [ImageToDepth](agentlego/tools/image_depth/README.md#ImageToDepth): 生成图像的深度图像。
- [ImageToScribble](agentlego/tools/image_scribble/README.md#ImageToScribble): 生成一张图像的涂鸦草图。
- [ObjectDetection](agentlego/tools/object_detection/README.md#ObjectDetection): 检测图像中的所有物体。
- [TextToBbox](agentlego/tools/object_detection/README.md#TextToBbox): 检测图像中的给定对象。
- Segment Anything 系列工具
  - [SegmentAnything](agentlego/tools/segmentation/README.md#SegmentAnything): 分割图像中的所有物体。
  - [SegmentObject](agentlego/tools/segmentation/README.md#SegmentObject): 根据给定的物体名称，在图像中分割出特定的物体。

**AIGC 相关**

- [TextToImage](agentlego/tools/image_text/README.md#TextToImage): 根据输入文本生成一张图片。
- [ImageExpansion](agentlego/tools/image_editing/README.md#ImageExpansion): 根据图像的内容扩展图像的周边区域。
- [ObjectRemove](agentlego/tools/image_editing/README.md#ObjectRemove): 删除图像中的特定对象。
- [ObjectReplace](agentlego/tools/image_editing/README.md#ObjectReplace): 替换图像中的特定对象。
- [ImageStylization](agentlego/tools/image_editing/README.md#ImageStylization): 根据指令修改一张图片。
- ControlNet 系列工具
  - [CannyTextToImage](agentlego/tools/image_canny/README.md#CannyTextToImage): 根据 Canny 边缘图像和描述生成图像。
  - [DepthTextToImage](agentlego/tools/image_depth/README.md#DepthTextToImage): 根据深度图像和描述生成图像。
  - [PoseToImage](agentlego/tools/image_pose/README.md#PoseToImage): 根据人体姿势图像和描述生成图像。
  - [ScribbleTextToImage](agentlego/tools/image_scribble/README.md#ScribbleTextToImage): 根据涂鸦草图和描述生成图像。
- ImageBind 系列工具
  - [AudioToImage](agentlego/tools/imagebind/README.md#AudioToImage): 根据音频生成图像。
  - [ThermalToImage](agentlego/tools/imagebind/README.md#ThermalToImage): 根据热成像图生成一张图像。
  - [AudioImageToImage](agentlego/tools/imagebind/README.md#AudioImageToImage): 根据音频和图像生成新的图像。
  - [AudioTextToImage](agentlego/tools/imagebind/README.md#AudioTextToImage): 从音频和文本提示生成图像。

## 开源许可证

该项目采用[Apache 2.0 开源许可证](LICENSE)。
