import json
import time
from queue import Queue
from threading import Thread
from typing import Iterator, List, Optional, Union

import langchain_core.messages as lc_msg
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder, PromptTemplate,
                               SystemMessagePromptTemplate)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI as _ChatOpenAI
from pydantic import BaseModel

from agentlego.tools import BaseTool
from .. import message_schema as msg
from ..logging import logger
from ..ui import get_translator
from ..utils import parse_inputs, parse_outputs

i18n = get_translator(__file__)

# modified form hub.pull("hwchase17/structured-chat-agent")
STRUCTURED_CHAT_PROMPT = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools', 'meta_prompt'],
    input_types={
        'chat_history':
        List[Union[lc_msg.AIMessage, lc_msg.HumanMessage,
                   lc_msg.ChatMessage, lc_msg.SystemMessage,
                   lc_msg.FunctionMessage, lc_msg.ToolMessage]],
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['tool_names', 'tools'],
                template=
                'Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tools}\n\nUse a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid "action" values: "Final Answer" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}```\n\nBegin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. The thought and final answer should use the same language with the question. Format is Action:```$JSON_BLOB```then Observation'
            )),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['agent_scratchpad', 'input'],
                template=
                '{input}\n\n{agent_scratchpad}\n (reminder to respond in a JSON blob no matter what)'
            ))
    ])

class StopChainException(Exception):
    """Stop the chain by user."""


class GenerationCallback(BaseCallbackHandler):
    raise_error: bool = True

    def __init__(self, mq: Queue, tools: List[BaseTool]):
        self.mq = mq
        self.tools = {tool.name: tool for tool in tools}

    def on_agent_action(self, action: AgentAction, **kwargs):
        if 'Thought:' in action.log:
            thought = action.log.partition('Thought:')[-1].partition('\n')[0].strip()
        else:
            thought = None
        tool = self.tools[action.tool]
        args = parse_inputs(tool.toolmeta, action.tool_input)
        self.mq.put(msg.ToolInput(name=action.tool, args=args, thought=thought))

    def on_tool_end(self, output: str, name: str, **kwargs):
        if name in self.tools:
            tool = self.tools[name]
            # Try to parse the outputs
            outputs = parse_outputs(tool.toolmeta, output)
            self.mq.put(msg.ToolOutput(outputs=outputs))
        else:
            self.mq.put(msg.ToolOutput(error=output))

    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        self.mq.put(msg.Answer(text=finish.return_values['output']))

    def on_tool_start(self, *args, **kwargs):
        from .. import shared
        if shared.stop_everything:
            raise StopChainException('The chain is stopped by user.')

    def on_llm_start(self, serialized, prompts, *args, **kwargs):
        from .. import shared
        if shared.stop_everything:
            raise StopChainException('The chain is stopped by user.')
        if shared.args.verbose:
            logger.info('LangChain prompt: \n' + '\n'.join(prompts))

    def on_chain_error(self, error: BaseException, **kwargs):

        self.mq.put(msg.Error(type=type(error).__name__, reason=str(error)))


class ChatOpenAI(_ChatOpenAI):
    """Support Extra stop words."""
    extra_stop: Optional[List[str]] = None

    def _create_message_dicts(self, messages, stop):
        if stop is not None and self.extra_stop is not None:
            stop = stop + self.extra_stop
        elif stop is None and self.extra_stop is not None:
            stop = self.extra_stop
        return super()._create_message_dicts(messages, stop=stop)

def llm_chat_openai(cfg):

    extra_stop = cfg.get('extra_stop')
    if isinstance(extra_stop, str) and len(extra_stop) > 0:
        extra_stop = extra_stop.split(',')
    else:
        extra_stop = None

    openai_api_base = cfg.get('openai_api_base')
    openai_api_key = cfg.get('openai_api_key')
    if openai_api_base and not openai_api_key:
        # Set a dummy key for self-hosted LLM
        openai_api_key = 'DUMMY_API_KEY'

    llm = ChatOpenAI(
        model=cfg.get('model_name'),
        max_tokens=cfg.get('max_tokens'),
        base_url=openai_api_base,
        api_key=openai_api_key,
        temperature=cfg.get('temperature', 0.7),
        extra_stop=extra_stop,
    )

    return llm

def cfg_chat_openai():
    import gradio as gr
    widgets = {}
    widgets['model_name'] = gr.Textbox(label='Model name')
    widgets['openai_api_base'] = gr.Textbox(label='API base url', info=i18n('openai_api_base'))
    widgets['openai_api_key'] = gr.Textbox(label='API key', info=i18n('openai_api_key'))
    widgets['max_tokens'] = gr.Slider(label='Max number of tokens', minimum=0, maximum=8192, step=256, value=512, info=i18n('max_tokens'))
    widgets['temperature'] = gr.Slider(label='Temperature', minimum=0., maximum=1., step=0.1, value=0.7, info=i18n('temperature'))
    widgets['extra_stop'] = gr.Textbox(label='Extra stop words', info=i18n('extra_stop'))
    widgets['meta_prompt'] = gr.Textbox(label='Meta prompt', info=i18n('meta_prompt'), value='Respond to the human as helpfully and accurately as possible.', lines=5)
    widgets['greeting'] = gr.Textbox(label='Greeting', info=i18n('greeting'), value=None, lines=5)
    return widgets


def langchain_style_history(history) -> ChatMessageHistory:
    memory = ChatMessageHistory()
    for row in history['internal']:
        response = ''
        if row[0]:
            memory.add_user_message(row[0])
        for step in row[1]:
            if isinstance(step, msg.ToolInput):
                if step.thought:
                    response += f'Thought: {step.thought}\n'
                args = json.dumps({k: v['content'] for k, v in step.args.items()}, ensure_ascii=False)
                tool_str = f'{{\n  "action": "{step.name}",\n  "action_input": {args}\n}}'
                response += 'Action:\n```\n' + tool_str + '\n```\n'
            elif isinstance(step, msg.ToolOutput):
                if step.outputs:
                    outputs = ', '.join(out['content'] for out in step.outputs)
                    response += f'Observation: {outputs}\n'
                elif step.error:
                    response += f'Observation: {step.error}\n'
            elif isinstance(step, msg.Answer):
                response += f'Thought: {step.thought or ""}\n'
                tool_str = f'{{\n  "action": "Final Answer",\n  "action_input": "{step.text}"\n}}'
                response += 'Action:\n```\n' + tool_str + '\n```\n'
        memory.add_ai_message(response)
    return memory


def create_langchain_structure(llm, tools):
    from .. import shared

    tools = [tool.to_langchain() for tool in tools]
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=STRUCTURED_CHAT_PROMPT,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=shared.args.verbose,
        handle_parsing_errors=False,
    )


def generate_structured(question, state, history) -> Iterator[List[BaseModel]]:
    from .. import shared
    cfg = shared.agents_settings[shared.agent_name]
    messages = []

    mq = Queue()
    tools = [
        tool for k, tool in shared.toolkits.items()
        if k in state['selected_tools']
    ]
    callback = GenerationCallback(mq, tools)
    agent = create_langchain_structure(shared.llm, tools)

    history = langchain_style_history(history).messages
    if cfg.get('meta_prompt'):
        history = [lc_msg.HumanMessage(content=cfg['meta_prompt'])] + history

    thread = Thread(
        target=agent.invoke,
        args=(dict(input=question, chat_history=history), dict(callbacks=[callback], )))
    thread.start()
    while thread.is_alive() or mq.qsize() > 0:
        if mq.qsize() > 0:
            item = mq.get()
            messages.append(item)
            yield messages
            if isinstance(item, msg.Error):
                return
        elif shared.stop_everything:
            yield messages
            return
        else:
            time.sleep(0.5)
