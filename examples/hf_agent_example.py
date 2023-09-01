from pathlib import Path

from huggingface_hub import login
from transformers import HfAgent

from mmlmtools.apis.agents.transformers_agent import load_tools_for_hfagent

login()

tools = load_tools_for_hfagent(
    [
        'ImageCaption',
        'TextToSpeech',
    ],
    device='cpu',
)
agent = HfAgent(
    'https://api-inference.huggingface.co/models/bigcode/starcoder',
    additional_tools=tools)
# Remove the huggingface tools and only reserve mmtools.
for k in list(agent.toolbox.keys()):
    if agent.toolbox[k] not in tools:
        agent.toolbox.pop(k)

demo_img = (Path(__file__).parent / 'demo.png').absolute()

user = f'Please tell me the description of `{demo_img}`'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))

print(' -------------------- ')
user = 'Please speak the above description into audio'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))
