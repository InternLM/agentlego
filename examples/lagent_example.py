from lagent.actions.action_executor import ActionExecutor
from lagent.agents.react import ReAct
from lagent.llms.openai import GPTAPI
from prompt_toolkit import ANSI, prompt

from agentlego.apis import load_tool


def main():
    # set OPEN_API_KEY in your environment or directly pass it with key=''
    model = GPTAPI(model_type='gpt-3.5-turbo')

    tools = [
        load_tool(tool_type).to_lagent() for tool_type in [
            'GoogleSearch',
            'Calculator',
        ]
    ]
    chatbot = ReAct(
        llm=model,
        max_turn=3,
        action_executor=ActionExecutor(actions=tools),
    )
    system = chatbot._protocol.format([], [],
                                      chatbot._action_executor)[0]['content']
    print(f'\033[92mSystem\033[0m:\n{system}')

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if user == 'exit':
            exit(0)

        try:
            chatbot.chat(user)
        finally:
            for history in chatbot._inner_history[1:]:
                if history['role'] == 'system':
                    print(f"\033[92mSystem\033[0m:{history['content']}")
                elif history['role'] == 'assistant':
                    print(f"\033[92mBot\033[0m:\n{history['content']}")


if __name__ == '__main__':
    main()
