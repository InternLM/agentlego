import copy
import json
from typing import Any, Iterator, List, Union

import requests
from lagent.actions import ActionExecutor
from lagent.agents import internlm2_agent
from lagent.llms.base_llm import BaseModel
from lagent.llms.meta_template import INTERNLM2_META
from lagent.schema import AgentStatusCode, ModelStatusCode
from lagent.utils.util import filter_suffix

from .. import message_schema as msg
from ..logging import logger
from ..ui import get_translator
from ..utils import parse_inputs

i18n = get_translator(__file__)


class LMDeployClient(BaseModel):
    """An LMDeploy client without lmdeploy dependency."""
    def __init__(self, url: str, model_name: str, **kwargs):
        BaseModel.__init__(self, path=url, **kwargs)
        self.completions_v1_url = f'{url}/v1/completions'
        self.model_name = model_name

    def stream_chat(self,
                    inputs: List[dict],
                    session_id=0,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    ignore_eos: bool = False,
                    timeout: int = 30,
                    **kwargs):
        gen_params = self.update_gen_params(**kwargs)
        prompt = self.template_parser(inputs)

        resp = ''
        finished = False
        stop_words = self.gen_params.get('stop_words')
        for text in self.stream_completions(
                model=self.model_name,
                prompt=prompt,
                session_id=session_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                ignore_eos=ignore_eos,
                timeout=timeout,
                **gen_params):
            resp += text['choices'][0]['text']
            if not resp:
                continue
            # remove stop_words
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

    def stream_completions(self, model: str, prompt: Union[str, List[Any]], **kwargs):
        payload = {
            'model': model,
            'prompt': prompt,
            'suffix': None,
            'temperature': 0.7,
            'n': 1,
            'max_tokens': 16,
            'stop': None,
            'top_p': 1.0,
            'top_k': 40,
            'user': None,
            'repetition_penalty': 1.0,
            'session_id': -1,
            'ignore_eos': False,
            'stream': True,
            **kwargs,
        }
        headers = {'content-type': 'application/json'}
        response = requests.post(
            self.completions_v1_url, headers=headers, json=payload, stream=True)
        for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded == 'data: [DONE]':
                    continue
                if decoded[:6] == 'data: ':
                    decoded = decoded[6:]
                try:
                    output = json.loads(decoded)
                    yield output
                except json.JSONDecodeError:
                    logger.warning(f'weird json content {decoded}')


def llm_internlm2_lmdeploy(cfg):
    url = cfg['url'].strip()
    llm = LMDeployClient(
        model_name='internlm2-chat-20b',
        url=url,
        meta_template=INTERNLM2_META,
        top_p=0.8,
        top_k=100,
        temperature=cfg.get('temperature', 0.7),
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])
    return llm


def cfg_internlm2():
    import gradio as gr
    widgets = {}
    widgets['url'] = gr.Textbox(label='URL', info=i18n('url'))
    widgets['max_turn'] = gr.Slider(label=i18n('max_turn'), value=6, minimum=1, maximum=12, step=1)
    widgets['temperature'] = gr.Slider(label='Temperature', minimum=0., maximum=1., step=0.1, value=0.7, info=i18n('temperature'))
    widgets['meta_prompt'] = gr.Textbox(label='System prompt', value=internlm2_agent.META_CN, lines=5)
    widgets['plugin_prompt'] = gr.Textbox(label='Plugin prompt', value=internlm2_agent.PLUGIN_CN, lines=5)
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
        protocol=internlm2_agent.Internlm2Protocol(
            meta_prompt=cfg['meta_prompt'].strip(),
            plugin_prompt=cfg['plugin_prompt'].strip(),
            tool=dict(
                begin='{start_token}{name}\n',
                start_token='<|action_start|>',
                name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),
                belong='assistant',
                end='<|action_end|>\n',
            ),
        ),
        max_turn=cfg.get('max_turn', 6),
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
