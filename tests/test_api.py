from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import pytest

from mmlmtools.api import (MMTOOLS, Mode, ToolMeta, collect_tools, load_tool,
                           register_custom_tool)
from mmlmtools.utils import _get_required_repos

REPOS = _get_required_repos()


def skip_test():
    for repo in REPOS:
        if find_spec(repo) is None:
            return True
    return False


# TODO Remove patch
def test_load_tool():
    patched_default_tools = {
        'object detection': {
            'efficiency': ToolMeta(MagicMock(), 'yolov5_tiny'),
            'balance': ToolMeta(MagicMock(), 'yolov5_s'),
            'performance': ToolMeta(MagicMock(), 'yolov5_l')
        },
        'image classification': ToolMeta(MagicMock(), 'resnet')
    }

    patched_tools = {'object detection': {'ssd': ToolMeta(MagicMock(), 'ssd')}}
    with patch('mmlmtools.api.DEFAULT_TOOLS', patched_default_tools), \
         patch('mmlmtools.api.MMTOOLS', patched_tools):
        # Catch exception
        with pytest.raises(ValueError, match='available tools are'):
            load_tool('unknown')

        with pytest.raises(ValueError, match='available modes are'):
            load_tool('object detection', mode='unknown')

        with pytest.raises(ValueError, match='mode should not be configured'):
            load_tool('image classification', mode='efficiency')

        with pytest.raises(ValueError, match='available model names'):
            load_tool('object detection', model='unknown')

        with pytest.raises(
                ValueError, match='mode should not be configured when model'):
            load_tool(
                'image classification', model='resnet', mode='efficiency')

        # 1. Test load tool from DEFAULT_TOOLS
        det_tool, meta = load_tool('object detection')
        assert meta.model == 'yolov5_tiny'

        det_tool, meta = load_tool('object detection', mode='balance')
        assert meta.model == 'yolov5_s'

        det_tool, meta = load_tool('object detection', mode=Mode.performance)
        assert meta.model == 'yolov5_l'

        # return cached tool
        cached_tool, meta = load_tool(
            'object detection', mode=Mode.performance)
        assert cached_tool is det_tool

        cls_tool, meta = load_tool('image classification')
        assert meta.model == 'resnet'

        # 2. Test load tool from TOOLS
        det_tool, meta = load_tool('object detection', model='ssd')
        assert meta.model == 'ssd'


@pytest.mark.skipif(
    skip_test(), reason='Only test when all related repos is installed')
def test_collect_tools():
    mocked_task2tool = MagicMock()
    mocked_task2tool.__getitem__ = lambda *args, **kwarg: MagicMock
    mocked_task2tool.__contains__ = lambda *args, **kwarg: True
    with patch('mmlmtools.api.TASK2TOOL', mocked_task2tool):
        collect_tools()
        assert 'object detection' in MMTOOLS


def test_register_custom_tools():
    patched_default_tools = {
        'object detection': {
            'efficiency': ToolMeta(MagicMock(), 'yolov5_tiny'),
            'balance': ToolMeta(MagicMock(), 'yolov5_s'),
            'performance': ToolMeta(MagicMock(), 'yolov5_l')
        },
        'image classification': ToolMeta(MagicMock(), 'resnet')
    }
    with patch('mmlmtools.api.DEFAULT_TOOLS', patched_default_tools):

        @register_custom_tool(
            tool='code executor', description='python code executor')
        def python_code_exec1(inputs):
            return eval(inputs)

        tool, meta = load_tool('code executor')
        res = tool('1+1')
        assert res == 2
        assert meta.description == 'python code executor'

        # tool with duplicated name
        with pytest.raises(KeyError):

            @register_custom_tool(
                tool='code executor', description='python code executor')
            def python_code_exec2(inputs):
                return eval(inputs)

        @register_custom_tool(
            tool='code executor',
            description='python code executor',
            force=True)
        def python_code_exec3(inputs):
            return eval(inputs)
