import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import yaml
from modules.logging import logger

from agentlego.apis.tool import NAMES2TOOLS, extract_all_tools
from agentlego.tools import BaseTool
from agentlego.tools.remote import RemoteTool
from agentlego.utils import resolve_module

# Agent variables
agent_name = None
agent = None
llm = None
toolkits: Mapping[str, BaseTool] = {}

# Generation variables
stop_everything = False
need_restart = False
generation_lock = None
processing_message = '*Is thinking...*'

# UI variables
gradio = {}

# UI defaults
settings = {
    'preset': 'simple-1',
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'seed': -1,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 200000,
    'max_tokens_second': 0,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'add_bos_token': True,
    'skip_special_tokens': True,
    'stream': True,
    'autoload_model': False,
}

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))

# Basic settings
parser.add_argument('--agent', type=str, help='Name of the agent to load by default.')
parser.add_argument('--agent-config', type=str, default='agent_config.yml', help='The agent config yaml file.')
parser.add_argument('--tool-config', type=str, default='tool_config.yml', help='The tools config yaml file.')
parser.add_argument('--lazy-setup', action='store_true', help='Avoid setup tools before the first run.')
parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')

# Gradio
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
parser.add_argument('--gradio-auth', type=str, help='Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".', default=None)
parser.add_argument('--gradio-auth-path', type=str, help='Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.', default=None)
parser.add_argument('--ssl-keyfile', type=str, help='The path to the SSL certificate key file.', default=None)
parser.add_argument('--ssl-certfile', type=str, help='The path to the SSL certificate cert file.', default=None)

args = parser.parse_args()

# Security warnings
if args.share:
    logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")
if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
    logger.warning("You are potentially exposing the web UI to the entire internet without any access password.\nYou can create one with the \"--gradio-auth\" flag like this:\n\n--gradio-auth username:password\n\nMake sure to replace username:password with your own.")


def download_to_tmp_file(url, suffix=None):
    import hashlib
    from tempfile import gettempdir

    import requests
    content = requests.get(url).content
    path = Path(gettempdir()) / hashlib.sha256(content).hexdigest()[:8]
    if suffix:
        path = path.with_suffix(suffix)
    with open(path, 'wb') as f:
        f.write(content)
    return str(path)


# Load agent-specific settings
if args.agent_config.startswith('http'):
    args.agent_config = download_to_tmp_file(args.agent_config, suffix='.yml')
    logger.info(f'Downloaded the specified agent config to {args.agent_config}')
with Path(args.agent_config) as p:
    if p.exists():
        agents_settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        agents_settings = {}

agents_settings = OrderedDict(agents_settings)

# Load tool-specific settings
if args.tool_config.startswith('http'):
    args.tool_config = download_to_tmp_file(args.tool_config, suffix='.yml')
    logger.info(f'Downloaded the specified tool config to {args.tool_config}')
with Path(args.tool_config) as p:
    if p.exists():
        tool_settings = yaml.safe_load(open(p, 'r', encoding='utf-8').read())
    else:
        tool_settings = {}

tool_classes: Mapping[str, type] = NAMES2TOOLS.copy()
tool_classes['RemoteTool'] = RemoteTool
custom_tools_dir = Path(__file__).absolute().parents[1] / 'custom_tools'
for source_file in custom_tools_dir.glob('*.py'):
    try:
        module = resolve_module(source_file)
        toolkit = module.__name__
        if toolkit.startswith('_ext_'):
            toolkit = toolkit[5:]
        tool_classes.update({
            toolkit + '.' + k: v
            for k, v in extract_all_tools(module).items()
        })
    except Exception:
        logger.exception('Traceback')
