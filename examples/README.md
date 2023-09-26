## 测试指南

1. 安装工具依赖（暂时方案）

```bash
pip install openmim
mim install mmcv
mim install mmocr
mim install mmdet

git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip install -e .
cd ..

git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -e .
cd ..


git clone https://github.com/open-mmlab/mmpose.git -b dev-1.x
cd mmpose
pip install -e .
cd ..

mim install mmsegmentation

cd visual_chatgpt/

# create a new environment
conda create -n visgpt python=3.8

# activate the new environment
conda activate visgpt

#  prepare the basic environments
pip install -r requirements.txt

# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}
```

2. Quick Start

我们在 `examples/visual_chatgpt/` 下给了一个简化过的 visual chatgpt 代码。

在本例中，我们使用了 visual chatgpt 中原生支持的 `ImageCaptioning` 和 AgentLego 提供的 `OCRTool` 两个工具，来演示如何将 AgentLego 集成到已有的项目中。

```bash
python visual_chatgpt.py --load "ImageCaptioning_cpu,OCRTool_cpu"
```
