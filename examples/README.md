## 测试指南

1. 拉取 Visual ChatGPT 项目代码并安装

```bash
# clone the repo
git clone https://github.com/microsoft/TaskMatrix.git

# Go to directory
cd TaskMatrix

#  prepare the basic environments
pip install -r requirements.txt
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install  git+https://github.com/facebookresearch/segment-anything.git

# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}

```

2. 安装工具依赖（暂时方案）

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
```

3. 启动

我们在 examples/ 下给了一个简化过的 visual chatgpt 代码，可以直接运行。在本例中，我们使用了 visual chatgpt 中原生支持的 `ImageCaptioning` 和 mmtools 提供的 `OCRTool` 两个工具，来演示如何将 mmtools 集成到已有的项目中。

```bash
python visual_chatgpt.py --load "ImageCaptioning_cpu,OCRTool_cpu"
```
