import argparse

from lagent import GPTAPI, ActionExecutor, ReAct
from prompt_toolkit import ANSI, prompt

from agentlego.apis import load_tool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        default=['GoogleSearch', 'Calculator'],
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # set OPEN_API_KEY in your environment or directly pass it with key=''
    model = GPTAPI(model_type=args.model)
    tools = [load_tool(tool_type).to_lagent() for tool_type in args.tools]
    chatbot = ReAct(
        llm=model,
        max_turn=3,
        action_executor=ActionExecutor(actions=tools),
    )

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if user == 'exit':
            exit(0)

        result = chatbot.chat(user)
        for history in result.inner_steps:
            if history['role'] == 'system':
                print(f"\033[92mSystem\033[0m:{history['content']}")
            elif history['role'] == 'assistant':
                print(f"\033[92mBot\033[0m:\n{history['content']}")


if __name__ == '__main__':
    main()
