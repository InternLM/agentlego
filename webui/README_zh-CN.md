# AgentLego WebUI

一个可以方便和 Agent 系统对话的 Gradio WebUI

## 设置您的语言模型

AgentLego 和这个 WebUI 并不负责部署大语言模型。因此，您需要使用其他框架来托管大语言模型作为 Agent 系统的后端。

### OpenAI模型

一个简单的选择是使用 OpenAI 的 API，我们已经在预设配置中提供了使用 OpenAI 模型的 Agent 配置。

您需要在环境变量中设置 OpenAI API key。

```bash
export OPENAI_API_KEY="你的 OpenAI API key"
```

### LMDeploy

LMDeploy 是一个用于压缩、部署和提供大型语言模型(LLM)服务的工具包，由 **InternLM** 团队开发。

要使用 LMDeploy 托管一个 LLM，请使用以下命令安装 LMDeploy（更多详情请参见[官方教程](https://lmdeploy.readthedocs.io/en/latest/get_started.html)）：

```bash
pip install 'lmdeploy>=0.2.1'
```

然后，通过以下命令托管一个类 OpenAI 风格的 API 服务器，这里我们以 InternLM2 为例。

```bash
lmdeploy serve api_server internlm/internlm2-chat-20b
```

启动后，你将得到以下输出：

```
INFO:     Started server process [853738]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

### vLLM

vLLM 是一个快速且易于使用的大型语言模型(LLM)推理和服务库。

要使用 vLLM 托管一个 LLM，请使用以下命令安装 vLLM（更多详情请参见[官方教程](https://docs.vllm.ai/en/latest/getting_started/installation.html)）：

```bash
# (可选) 创建一个新的 conda 环境。
conda create -n myenv python=3.9 -y
conda activate myenv

# 安装带 CUDA 12.1 的 vLLM。
pip install vllm
```

然后，通过以下命令托管一个类 OpenAI 风格的 API 服务器，这里我们以 QWen 为例。

```bash
# 获取 ChatML 风格的聊天模板
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/template_chatml.jinja

# 启动 vLLM 服务器
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-14B-Chat --trust-remote-code --chat-template ./template_chatml.jinja
```

启动后，你将得到以下输出：

```
INFO:     Started server process [3837676]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 设置 WebUI

你可以使用 `start_linux.sh` 从头开始创建一个独立的环境，或者使用 `one_click.py` 在你自己的环境上设置环境。

```bash
bash startup_linux.sh
# 或者
python one_click.py
```

然后你将得到以下输出，之后，在浏览器中打开该 URL。

```
Running on local URL:  http://127.0.0.1:7860
```

## 与 Agent 聊天

打开 WebUI 后，你需要在 `Agent` 标签中选择 Agent。如果你部署了自己的 LLM，可以使用 `langchain.StructuredChat.ChatOpenAI`，并在 `API base url` 字段中传递你的 LLM 服务器的 URL。

要设置可用的工具，你需要在 `Tools` 标签中添加它们并设置 `enable`。

如果 Agent 的响应引发了解析错误（对于执行指令能力较低的 LLM 来说很常见）或你想重新生成响应，你可以点击 `Regenerate` 以重新生成最后一个响应。

在聊天过程中，你可以从所有可用的工具中选择工具，勾选 `Select tools` 复选框。

如果你想保存当前聊天，点击 `Save` 按钮。你也可以从 `Past chats` 下拉菜单中恢复过去的聊天。

在聊天过程中，你可以上传文件（图片或音频）。如果你已提供相关工具，Agent 可能会使用这些工具来处理你的文件。

## 致谢

WebUI 应用是基于 [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) 修改的。感谢他们的出色工作。
