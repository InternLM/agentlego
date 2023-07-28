import os.path as osp
import tempfile
from importlib.util import find_spec

import pytest
from mmengine.hub import get_config

from mmlmtools.apis.tool import custom_tool, load_tool
from mmlmtools.tools import ImageCaptionTool

REPOS = ['mmpretrain', 'mmdet', 'mmagic']


def skip_test():
    for repo in REPOS:
        if find_spec(repo) is None:
            return True
    return False


# TODO Remove patch
@pytest.mark.skipif(
    skip_test(), reason='Only test when all related repos is installed')
def test_load_tool():
    # load tool without with default model
    tool = load_tool('ImageCaptionTool')
    assert isinstance(tool, ImageCaptionTool)

    # load tool from local cfg
    cfg = get_config('mmpretrain::blip/blip-base_8xb32_caption.py')
    with tempfile.TemporaryDirectory() as tempdir:
        cfg_path = osp.join(tempdir, 'config.py')
        cfg.dump(cfg_path)
        tool = load_tool('ImageCaptionTool', model=cfg_path)
        assert isinstance(tool, ImageCaptionTool)

    # load tool from specified model
    tool = load_tool('ObjectDetectionTool', model='atss_r50_fpn_1x_coco')
    assert tool.toolmeta.model == 'atss_r50_fpn_1x_coco'

    # description will be overwrite
    tool = load_tool('Canny2ImageTool', description='custom')
    assert 'custom' in tool.toolmeta.description

    tool = load_tool('Canny2ImageTool', input_description='custom')
    assert 'custom' in tool.toolmeta.input_description

    tool = load_tool('Canny2ImageTool', output_description='custom')
    assert 'custom' in tool.toolmeta.output_description

    # cached tool
    cached_tool = tool = load_tool(
        'Canny2ImageTool', output_description='custom')
    assert cached_tool is tool


def test_register_custom_tools():

    @custom_tool(
        tool_name='code executor',
        description='python code executor',
        output_description='nothing',
        input_description='nothing')
    def python_code_exec1(inputs):
        return eval(inputs)

    tool = load_tool('code executor')
    res = tool('1+1')
    assert res == 2
    assert tool.toolmeta.description == 'python code executor'
    assert tool.toolmeta.input_description == 'nothing'
    assert tool.toolmeta.output_description == 'nothing'

    # tool with duplicated name
    with pytest.raises(KeyError):

        @custom_tool(
            tool_name='code executor', description='python code executor')
        def python_code_exec2(inputs):
            return eval(inputs)

    @custom_tool(
        tool_name='code executor',
        description='python code executor',
        force=True)
    def python_code_exec3(inputs):
        return eval(inputs)
