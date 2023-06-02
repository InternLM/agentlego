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

git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip install -e .
cd ..

git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -e .
cd ..

git clone https://github.com/open-mmlab/mmdetection.git -b dev-3.x
cd mmdetection
pip install -e .
cd ..

git clone https://github.com/open-mmlab/mmpose.git -b dev-1.x
cd mmpose
pip install -e .
cd ..

mim install mmseg
```

3. 用本目录下的 `visual_chatgpt_XXX.py` 覆盖官方 `TaskMatrix/` 下的同名文件

4. 启动

```bash
cd TaskMatrix

python visual_chatgpt.py
```
