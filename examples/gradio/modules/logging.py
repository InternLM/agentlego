import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logger = logging.getLogger('agentlego')

handler = RichHandler(console=console, keywords=[], show_path=False)
handler.setFormatter(logging.Formatter('%(message)s', datefmt='[%X]'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
