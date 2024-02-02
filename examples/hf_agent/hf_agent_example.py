from pathlib import Path

from transformers import HfAgent

from agentlego.apis import load_tool

#  from huggingface_hub import login
#  login()

tools = [
    load_tool(tool_type).to_transformers_agent() for tool_type in [
        'ImageDescription',
        'TextToSpeech',
    ]
]
agent = HfAgent(
    'https://api-inference.huggingface.co/models/bigcode/starcoder',
    chat_prompt_template=(Path(__file__).parent / 'hf_demo_prompts.txt').read_text(),
    additional_tools=tools,
)

# Remove default tools in the transformers agent and only keep the tools from
# AgentLego. Please note that this is only for demo purpose. In practice,
# AgentLego tools can be used together with other tools.

for k in list(agent.toolbox.keys()):
    if agent.toolbox[k] not in tools:
        agent.toolbox.pop(k)

demo_img = Path(__file__).absolute().parents[1] / 'demo.png'

user = f'Describe the image `{demo_img}` and save to variable `description`.'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))

print(' -------------------- ')
user = 'Please speak the above description into audio'
print(f'\033[92mUser\033[0m: {user}')
print('\033[92mBot\033[0m:', agent.chat(user))
