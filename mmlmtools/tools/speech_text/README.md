# TextToSpeech

```{eval-rst}
.. autoclass:: mmlmtools.tools.TextToSpeech
    :noindex:
```

## Default Tool Meta

- **name**: Text Reader
- **description**: This is a tool that can speak the input English text into audio.
- **inputs**: text
- **outputs**: audio

## Examples

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('TextToSpeech', device='cuda')

# apply tool
audio = tool('Hello, this is a text to audio demo.')
print(audio)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['TextToSpeech'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'Please introduce the highest mountain and speak out.')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U transformers
```

## Reference

This tool uses a **Speech T5** model in default settings. See the following paper for details.

```bibtex
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2021}
}
```

# SpeechToText

```{eval-rst}
.. autoclass:: mmlmtools.tools.SpeechToText
    :noindex:
```

## Default Tool Meta

- **name**: Transcriber
- **description**: This is a tool that transcribes an audio into text.
- **inputs**: audio
- **outputs**: text

## Examples

**Use the tool directly (without agent)**

```python
from mmlmtools.apis import load_tool

# load tool
tool = load_tool('SpeechToText', device='cuda')

# apply tool
text = tool('examples/demo.m4a')
print(text)
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from mmlmtools.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['SpeechToText'], device='cuda')
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
audio_path = 'examples/demo.m4a'
ret = agent.chat(f'Please tell me the content of the audio `{audio_path}`')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
pip install -U transformers
```

## Reference

This tool uses a **Whisper** model in default settings. See the following paper for details.

```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
