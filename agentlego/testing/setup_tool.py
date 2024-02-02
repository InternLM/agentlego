import os
from urllib.parse import quote_plus, urljoin

from agentlego.tools.remote import RemoteTool


def setup_tool(tool_type, **kwargs):
    remote_url = os.getenv('AGENTLEGO_SERVER', None)
    if not remote_url:
        return tool_type(**kwargs)
    else:
        domain = quote_plus(tool_type.DEFAULT_TOOLMETA.name.replace(' ', ''))
        return RemoteTool.from_url(urljoin(remote_url, domain))
