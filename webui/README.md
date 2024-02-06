# AgentLego WebUI

An easy-to-use Gradio App to setup agent and tools.

## Setup your LLM

AgentLego and this web ui doesn't aim to host LLM, therefore, you need to use other framework to host a LLM as
the backend of agents.

### OpenAI models

A simple choice is to use OpenAI's models, and we have provided the agent configs to use OpenAI models in the
preset configs. You need to set the OpenAI API key in the environment variables.

```bash
export OPENAI_API_KEY="your openai key"
```

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the **InternLM** teams.

To host a LLM with LMDeploy, use the below command to install LMDeploy (See the [official tutorial](https://lmdeploy.readthedocs.io/en/latest/get_started.html) for more details):

```bash
pip install 'lmdeploy>=0.2.1'
```

And then, host an OpenAI-style API server by the below command, here we use InternLM2 as example.

```bash
lmdeploy serve api_server internlm/internlm2-chat-20b
```

And after startup, you will get the below output:

```
INFO:     Started server process [853738]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

### vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.

To host a LLM with vLLM, use the below command to install vLLM (See the [official tutorial](https://docs.vllm.ai/en/latest/getting_started/installation.html) for more details):

```bash
# (Optional) Create a new conda environment.
conda create -n myenv python=3.9 -y
conda activate myenv

# Install vLLM with CUDA 12.1.
pip install vllm
```

And then, host an OpenAI-style API server by the below command, here we use QWen as example.

```bash
# Get the ChatML style chat template
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/template_chatml.jinja

# Start the vLLM server
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-14B-Chat --trust-remote-code --chat-template ./template_chatml.jinja
```

And after startup, you will get the below output:

```
INFO:     Started server process [3837676]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Setup WebUI

You can use the `start_linux.sh` or `start_windows.bat` to create a standalone environment from scratch, or use `one_click.py` to
setup the environment on your own environment.

```bash
# On Windows, from scratch

bash startup_linux.sh
# OR
python one_click.py
```

Then you will get the below output and then, open the URL in your browser.

```
Running on local URL:  http://127.0.0.1:7860
```

## Chat with Agent

After open the web ui, you need to choose the agent in the `Agent` tab. If you are hosting your own LLM, you
can use `langchain.StructuredChat.ChatOpenAI` agent and pass the URL of your LLM server in the `API base url`
field.

And to setup available tools, you need to add them in the `Tools` tab.

If the response of agent raise an parse error (it's common for the LLM with low instruction-following ability)
or you want to re-roll the response, you can click `Regenerate` to regenerate the last response.

During chat, you can select tools from the all available tools in the `Select tools` checkboxes.

If you want to save the current chat, click the `Save` button. And you can also resume the past chats from the
`Past chats` dropdown.

You can upload files (images or audios) during chat. If you have provided related tools, agent may use these
tools to handle your file.

## Acknowledge

The WebUI app is modified from [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui). Thanks for the great work.
