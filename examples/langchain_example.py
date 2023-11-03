from langchain.agents.initialize import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from prompt_toolkit import ANSI, prompt

from agentlego.apis.agents.langchain import load_tools_for_langchain


def main():
    tools = load_tools_for_langchain([
        'ImageCaption',
        'Calculator',
        'SpeechToText',
    ])
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent='chat-conversational-react-description',
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

        print(agent.invoke({'input': user})['output'])


if __name__ == '__main__':
    main()
