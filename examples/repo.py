import argparse

from langchain.agents import Tool

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent

from mmlmtools.apis.agents.visual_chatgpt import load_tools_for_visual_chatgpt


if __name__ == '__main__':
    # load our tools
    globals().update(load_tools_for_visual_chatgpt())
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='ImageCaptionTool_cuda:0')
    args = parser.parse_args()

    # get the tools we want to load
    load_dict = {
        e.split('_')[0].strip(): e.split('_')[1].strip()
        for e in args.load.split(',')
    }

    print(f'Initializing tools,load_dict={load_dict}')
    models = {}
    for class_name, device in load_dict.items():
        models[class_name] = globals()[class_name](device=device)

    print(f'All the Available Functions: {models}')

    tools = []
    for instance in models.values():
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)
                tools.append(
                    Tool(
                        name=func.name,
                        description=func.description,
                        func=func))
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', output_key='output')

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent='conversational-react-description',
        verbose=True,
        memory=memory
    )

    # give your input here
    input_path = ''
    agent.run(input=input_path)
