from pathlib import Path

from transformers import HfAgent

from agentlego.apis.agents.huggingface_agent import load_tools_for_hfagent

#  from huggingface_hub import login
#  login()

tools = load_tools_for_hfagent(
    [
        'ImageCaption',
        'TextToSpeech',
    ],
    device='cpu',
)
agent = HfAgent(
    'https://api-inference.huggingface.co/models/bigcode/starcoder',
    chat_prompt_template=(Path(__file__).parent /
                          'hf_demo_prompts.txt').read_text(),
    additional_tools=tools,
)

# Remove default tools in the huggingface agent and only keep the tools from
# AgentLego. Please note that this is only for demo purpose. In practice,
# AgentLego tools can be used together with other tools.

for k in list(agent.toolbox.keys()):
    if agent.toolbox[k] not in tools:
        agent.toolbox.pop(k)

demo_img = (Path(__file__).parent / 'demo.png').absolute()

user = f'Describe the image `{demo_img}` and save to variable `description`.'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))

print(' -------------------- ')
user = 'Please speak the above description into audio'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))
