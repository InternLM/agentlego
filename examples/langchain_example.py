import argparse

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
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

    tools = [load_tool(tool_type).to_langchain() for tool_type in args.tools]
    # set OPEN_API_KEY in your environment or directly pass it with key=''
    llm = ChatOpenAI(temperature=0, model=args.model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    agent = create_structured_chat_agent(
        llm,
        tools,
        prompt=hub.pull('hwchase17/structured-chat-agent')
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if user == 'exit':
            exit(0)
        res = agent_executor.invoke(dict(input=user))
        print(f'\033[91m{args.model}\033[0m: {res["output"]}')


if __name__ == '__main__':
    main()
