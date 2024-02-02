# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import random
import re
import uuid

import gradio as gr
import numpy as np
import torch
from langchain.agents.initialize import initialize_agent
from langchain.llms.openai import OpenAI
from langchain.memory.buffer import ConversationStringBufferMemory
from PIL import Image

from agentlego.apis import load_tool

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""  # noqa: E501

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""  # noqa: E501

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human.
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""  # noqa: E501

VISUAL_CHATGPT_PREFIX_CN = """Visual ChatGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 Visual ChatGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

Visual ChatGPT 能够处理和理解大量文本和图像。作为一种语言模型，Visual ChatGPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，Visual ChatGPT可以调用不同的工具来间接理解图片。在谈论图片时，Visual ChatGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，Visual ChatGPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 Visual ChatGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 Visual ChatGPT 提供带有描述的新图形。描述帮助 Visual ChatGPT 理解这个图像，但 Visual ChatGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，Visual ChatGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

Visual ChatGPT 可以使用这些工具:"""  # noqa: E501

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""  # noqa: E501

VISUAL_CHATGPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为Visual ChatGPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对Visual ChatGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""  # noqa: E501

os.makedirs('image', exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):

    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f'history_memory:{history_memory}, n_tokens: {n_tokens}')
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class ConversationBot:

    def __init__(self, load_dict):
        # load_dict = {
        #     'OCRTool':'cuda:0',
        #     'ImageDescription':'cuda:1',...}
        print(f'Initializing VisualChatGPT, load_dict={load_dict}')

        if 'ImageDescription' not in load_dict:
            raise ValueError('You have to load ImageDescription as a '
                             'basic function for VisualChatGPT')

        self.models = {}
        # Load tools
        for class_name, device in load_dict.items():
            self.models[class_name] = load_tool(class_name, device=device).to_langchain()

        print(f'All the Available Functions: {self.models}')

        self.tools = []
        for tool in self.models.values():
            self.tools.append(tool)
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationStringBufferMemory(
            memory_key='chat_history', output_key='output')

    def init_agent(self, lang):
        self.memory.clear()  # clear previous history
        if lang == 'English':
            PREFIX = VISUAL_CHATGPT_PREFIX
            FORMAT_INSTRUCTIONS = VISUAL_CHATGPT_FORMAT_INSTRUCTIONS
            SUFFIX = VISUAL_CHATGPT_SUFFIX
            place = 'Enter text and press enter, or upload an image'
            label_clear = 'Clear'
        else:
            PREFIX = VISUAL_CHATGPT_PREFIX_CN
            FORMAT_INSTRUCTIONS = VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN
            SUFFIX = VISUAL_CHATGPT_SUFFIX_CN
            place = '输入文字并回车，或者上传图片'
            label_clear = '清除'
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent='conversational-react-description',
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                'prefix': PREFIX,
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            },
        )
        return gr.update(visible=True), gr.update(visible=False), gr.update(
            placeholder=place), gr.update(value=label_clear)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({'input': text.strip()})
        res['output'] = res['output'].replace('\\', '/')
        response = re.sub('(image/[-\\w]*.png)',
                          lambda m: f'![](file={m.group(0)})*{m.group(0)}*',
                          res['output'])
        state = state + [(text, response)]

        return state, state

    def run_image(self, image, state, txt, lang):
        image_filename = os.path.join('image', f'{str(uuid.uuid4())[:8]}.png')
        print('======>Auto Resize Image...')
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, 'PNG')
        print(f'Resize image form {width}x{height} to {width_new}x{height_new}')
        description = self.models['ImageDescription'](image_filename)
        if lang == 'Chinese':
            Human_prompt = (f'\nHuman: 提供一张名为 {image_filename}的图片。'
                            f'它的描述是: {description}。 '
                            f'这些信息帮助你理解这个图像，'
                            f'但是你应该使用工具来完成下面的任务，'
                            f'而不是直接从我的描述中想象。 '
                            f'如果你明白了, 说 \"收到\". \n')
            AI_prompt = '收到。  '
        else:
            Human_prompt = (f'\nHuman: provide a figure named '
                            f'{image_filename}. '
                            f'The description is: {description}. '
                            f'This information helps you to understand '
                            f'this image, but you should use tools to '
                            f'finish following tasks, rather than directly '
                            f'imagine from my description. If you understand, '
                            f'say \"Received\". \n')
            AI_prompt = 'Received.  '
        self.agent.memory.buffer = self.agent.memory.buffer + \
            Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f'![](file={image_filename})*{image_filename}*', AI_prompt)]
        return state, state, f'{txt} {image_filename} '


if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='ImageDescription_cuda:0')
    args = parser.parse_args()
    load_dict = {
        e.split('_')[0].strip(): e.split('_')[1].strip()
        for e in args.load.split(',')
    }
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css='#chatbot .overflow-y-auto{height:500px}') as demo:
        lang = gr.Radio(choices=['Chinese', 'English'], value=None, label='Language')
        chatbot = gr.Chatbot(elem_id='chatbot', label='Visual ChatGPT')
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder='Enter text and press enter, '
                    'or upload an image').style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button('Clear')
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label='🖼️', file_types=['image'])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: '', None, txt)
        btn.upload(bot.run_image, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name='0.0.0.0', server_port=7861)
