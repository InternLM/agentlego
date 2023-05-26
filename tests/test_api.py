from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import pytest

from mmlmtools.api import (MMTOOLS, REPOS, Mode, collect_tools, load_tool,
                           register_custom_tool)


def skip_test():
    for repo in REPOS:
        if find_spec(repo) is None:
            return True
    return False


# TODO Remove patch
def test_load_tool():
    patched_default_tools = {
        'object detection': {
            'efficiency': dict(description='yolov5_tiny'),
            'balance': dict(description='yolov5_s'),
            'performance': dict(description='yolov5_l')
        },
        'image classification': dict(description='resnet')
    }

    patched_task2tool = {
        'object detection': MagicMock(),
        'image classification': MagicMock(),
    }

    patched_tools = {'object detection': {'ssd': dict(description='ssd')}}
    with patch('mmlmtools.api.DEFAULT_TOOLS', patched_default_tools), \
         patch('mmlmtools.api.MMTOOLS', patched_tools), \
         patch('mmlmtools.api.TASK2TOOL', patched_task2tool):
        # Catch exception
        with pytest.raises(ValueError, match='available tools are'):
            load_tool('unknown')

        with pytest.raises(ValueError, match='available modes are'):
            load_tool('object detection', mode='unknown')

        with pytest.raises(ValueError, match='mode should not be configured '):
            load_tool('image classification', mode='efficiency')

        with pytest.raises(ValueError, match='available model names'):
            load_tool('object detection', model='unknown')

        with pytest.raises(ValueError, match='unknown is not available for'):
            load_tool('object detection', mode='unknown')

        with pytest.raises(
                ValueError, match='mode should not be configured when model'):
            load_tool('object detection', model='resnet', mode='efficiency')

        # 1. Test load tool from DEFAULT_TOOLS
        det_tool, meta = load_tool('object detection')
        assert meta.description == 'yolov5_tiny'
        assert meta.tool_name == 'object detection'

        det_tool, meta = load_tool('object detection', mode='balance')
        assert meta.description == 'yolov5_s'
        assert meta.tool_name == 'object detection 2'

        det_tool, meta = load_tool('object detection', mode=Mode.performance)
        assert meta.description == 'yolov5_l'
        assert meta.tool_name == 'object detection 3'

        # return cached tool
        cached_tool, meta = load_tool(
            'object detection', mode=Mode.performance)
        assert cached_tool is det_tool

        cls_tool, meta = load_tool('image classification')
        assert meta.description == 'resnet'

        # 2. Test load tool from TOOLS
        det_tool, meta = load_tool('object detection', model='ssd')
        assert meta.description == 'ssd'


@pytest.mark.skipif(
    skip_test(), reason='Only test when all related repos is installed')
def test_collect_tools():
    mocked_task2tool = MagicMock()
    mocked_task2tool.__getitem__ = lambda *args, **kwarg: MagicMock
    mocked_task2tool.__contains__ = lambda *args, **kwarg: True
    with patch('mmlmtools.api.TASK2TOOL', mocked_task2tool):
        collect_tools()
        assert 'object detection' in MMTOOLS
        assert 'rtmdet_l_8xb32-300e_coco' in MMTOOLS['object detection']
        assert 'rtmdet-l' in MMTOOLS['object detection']


def test_register_custom_tools():
    patched_default_tools = {
        'object detection': {
            'efficiency': dict(description='yolov5_tiny'),
            'balance': dict(description='yolov5_s'),
            'performance': dict(description='yolov5_l')
        },
        'image classification': dict(description='resnet')
    }

    patched_task2tool = {
        'object detection': MagicMock(),
        'image classification': MagicMock(),
    }

    with patch('mmlmtools.api.DEFAULT_TOOLS', patched_default_tools), \
         patch('mmlmtools.api.TASK2TOOL', patched_task2tool):

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
