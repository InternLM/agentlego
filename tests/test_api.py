import copy
from importlib.util import find_spec
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from mmlmtools.api import REPOS, TOOLS, _collect_tools
from mmlmtools.lmtools.base_tool import BaseTool


def skip_test():
    for repo in REPOS:
        if find_spec(repo) is None:
            return True
    return False


class FakeTool(BaseTool):
    @staticmethod
    def format_description(info):
        info = copy.deepcopy(info)
        if 'Metadata' not in info[0]:
            return info
        if 'Metrics' not in info[0]['Result']:
            return info
        if ('FLOPs' in info[0]['Metadata'] and
            'Top 1 Accuracy' in info[0]['Result']['Metrics']
        ):
            flops = [i['Metadata']['FLOPs'] for i in info]
            accs = [i['Result']['Metrics']['Top 1 Accuracy'] for i in info]
            efficient_info = info[flops.index(min(flops))]
            performance_info = info[accs.index(max(accs))]

            eff_desc = efficient_info.get('Description', '')
            eff_desc = f'{eff_desc}\nIt is a high efficiency tool'
            pef_desc = performance_info.get('Description', '')
            pef_desc = f'{pef_desc}\nIt is a high performance tool'

            efficient_info['Description'] = eff_desc
            performance_info['Description'] = pef_desc
        return info


# TODO Remove this patch
mocked_task2tool = MagicMock()
mocked_task2tool.__getitem__ = lambda *args, **kwarg: FakeTool
@pytest.mark.skipif(
    skip_test(),
    reason='Only test when all related repos is installed')
@patch('mmlmtools.api.task2tool', mocked_task2tool)
def test_collect_tools():
    _collect_tools()
    print(TOOLS)
    