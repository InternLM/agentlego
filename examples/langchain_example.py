from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from prompt_toolkit import ANSI, prompt

from agentlego.apis import load_tool


def main():
    tools = [
        load_tool(tool_type).to_langchain() for tool_type in [
            'Calculator',
            'GoogleSearch',
        ]
    ]
    llm = ChatOpenAI(temperature=0, model='gpt-4')
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if user == 'exit':
            exit(0)
        print(agent.run(input=user))


if __name__ == '__main__':
    main()
