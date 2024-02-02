import copy
from typing import Iterator, List

from lagent.actions import ActionExecutor
from lagent.agents import internlm2_agent
from lagent.llms.lmdepoly_wrapper import LMDeployClient
from lagent.llms.meta_template import INTERNLM2_META
from lagent.schema import AgentStatusCode

from .. import message_schema as msg
from ..logging import logger
from ..utils import parse_inputs


def llm_internlm2_lmdeploy(cfg):
    url = cfg['url'].strip()
    llm = LMDeployClient(
        path='internlm2-chat-20b',
        url=url,
        meta_template=INTERNLM2_META,
        top_p=0.8,
        top_k=100,
        temperature=0,
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])
    return llm


def cfg_internlm2():
    import gradio as gr
    widgets = {}
    widgets['url'] = gr.Textbox(label='URL', info='The internlm2 server url of LMDeploy, like `http://localhost:23333`')
    widgets['meta_prompt'] = gr.Textbox(label='system prompt', value=internlm2_agent.META_CN)
    widgets['plugin_prompt'] = gr.Textbox(label='plugin prompt', value=internlm2_agent.PLUGIN_CN)
    return widgets


def lagent_style_history(history) -> List[dict]:
    inner_steps = []
    for row in history['internal']:
        inner_steps.append(dict(role='user', content=row[0]))
        for step in row[1]:
            if isinstance(step, msg.ToolInput):
                args = {k: v['content'] for k, v in step.args.items()}
                inner_steps.append(dict(role='tool', name='plugin', content=dict(name=step.name, parameters=args)))
            elif isinstance(step, msg.ToolOutput) and step.outputs:
                outputs = '\n'.join(item['content'] if item['type'] ==
                                    'text' else f'[{item["type"]}]({item["content"]})'
                                    for item in step.outputs)
                inner_steps.append(dict(role='environment', content=outputs, name='plugin'))
            elif isinstance(step, msg.Answer):
                inner_steps.append(dict(role='language', content=step.text))
    return inner_steps


def create_internlm2_agent(llm, tools, cfg) -> internlm2_agent.Internlm2Agent:

    tools = [tool.to_lagent() for tool in tools]

    agent = internlm2_agent.Internlm2Agent(
        llm=llm,
        plugin_executor=ActionExecutor(actions=tools),
        protocol=internlm2_agent.Interlm2Protocol(
            plugin_prompt=cfg['plugin_prompt'].strip(),
            tool=dict(
                begin='{start_token}{name}\n',
                start_token='<|action_start|>',
                name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),
                belong='assistant',
                end='<|action_end|>\n',
            ),
        ),
        max_turn=6,
    )
    return agent


def generate_internlm2(question, state, history) -> Iterator[List[msg.Message]]:
    from .. import shared

    cfg = copy.deepcopy(shared.agents_settings[shared.agent_name])
    tools = [tool for k, tool in shared.toolkits.items() if k in state['selected_tools']]
    agent = create_internlm2_agent(shared.llm, tools, cfg)
    messages: List[msg.Message] = []
    history = lagent_style_history(history) + [dict(role='user', content=question)]
    if shared.args.verbose:
        for dialog in agent._protocol.format(inner_step=history, plugin_executor=agent._action_executor):
            logger.info(f'[{dialog["role"].upper()}]: {dialog["content"]}')
    for agent_return in agent.stream_chat(history):
        if agent_return.state == AgentStatusCode.PLUGIN_RETURN:
            action = agent_return.actions[-1]
            tool = shared.toolkits[action.type]
            args = parse_inputs(tool.toolmeta, action.args)
            messages.append(
                msg.ToolInput(name=action.type, args=args, thought=action.thought))
            messages.append(msg.ToolOutput(outputs=tuple(action.result)))
            yield messages
        elif agent_return.state == AgentStatusCode.END and isinstance(agent_return.response, str):
            messages.append(msg.Answer(text=agent_return.response))
            yield messages
        elif agent_return.state == AgentStatusCode.STREAM_ING:
            yield messages + [msg.Answer(text=agent_return.response)]
