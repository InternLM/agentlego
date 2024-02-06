import time

import gradio as gr
import modules.shared as shared

from .agents import agent_func_map, clear_cache
from .logging import logger
from .message_schema import Error


def generate_reply(*args, **kwargs):
    shared.generation_lock.acquire()
    try:
        yield from _generate_reply(*args, **kwargs)
    finally:
        shared.generation_lock.release()


def _generate_reply(question, state, history):

    # Find the appropriate generation function
    if shared.agent_name is None or shared.llm is None:
        logger.error('No agent is loaded! Select one in the Agent tab.')
        raise gr.Error('No agent is loaded! Select one in the Agent tab.')

    agent_class = shared.agents_settings[shared.agent_name]['agent_class']
    generate_func = agent_func_map[agent_class].generate

    if shared.args.verbose:
        logger.info(f'question:\n{question}\n--------------------\n')

    shared.stop_everything = False
    clear_cache()
    t0 = time.time()

    try:
        yield from generate_func(question, state, history)
    except Exception as e:
        yield Error(type=type(e).__name__, reason=str(e))
    finally:
        t1 = time.time()
        logger.info(f'Output generated in {(t1-t0):.2f} seconds.')


def stop_everything_event():
    shared.stop_everything = True
