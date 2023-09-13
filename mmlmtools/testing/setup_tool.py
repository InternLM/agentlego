# Copyright (c) OpenMMLab. All rights reserved.
import os
from urllib.parse import quote_plus, urljoin

from mmlmtools.tools.remote import RemoteTool


def setup_tool(tool_type, **kwargs):
    remote_url = os.getenv('MMTOOLS_SERVER', None)
    if not remote_url:
        return tool_type(**kwargs)
    else:
        domain = quote_plus(tool_type.DEFAULT_TOOLMETA.name.replace(' ', ''))
        return RemoteTool(urljoin(remote_url, domain))
