# Copyright (c) OpenMMLab. All rights reserved.
import enum
import os.path as osp
import re
from dataclasses import dataclass
from typing import List, Optional

from mmengine.config.utils import MODULE2PACKAGE


class Mode(enum.Enum):
    efficiency = 'high efficiency'
    balance = 'balance'
    performance = 'high performance'


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        tool_type (callable): class object to build the tool or callable tool.
        model (str, optional): model name for OpenMMLab tools.
            Defaults to None.
        description (str, optional): Description for tool. Defaults to None
    """
    tool_type: callable
    model: Optional[str] = None
    description: Optional[str] = None


def get_required_repos() -> List[str]:

    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-e '):
            yield None
        else:
            # Remove versioning from the package
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]
            yield parts[0]

    repos = []
    fname = osp.join(
        osp.dirname(__file__), '.mim', 'requirements', 'optional.txt')
    if not osp.exists(fname):
        fname = osp.join(
            osp.dirname(__file__), '..', 'requirements', 'optional.txt')
    mmrepos = set(MODULE2PACKAGE.values())
    for package in parse_require_file(fname):
        if package is None:
            continue
        if package not in mmrepos:
            continue
        repos.append(package)
    return repos
