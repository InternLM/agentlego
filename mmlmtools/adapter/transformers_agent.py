# Copyright (c) OpenMMLab. All rights reserved.
from transformers.tools import Tool

from mmlmtools import list_tools, load_tool


class ToolAdapter(Tool):
    """Adapter for mmlmtools.tools.Tool to transformers.tools.Tool."""

    def __init__(self, tool):
        self.name = tool.toolmeta.name
        self.tool = tool

        # Transformers Agent requires the input or output to be a PIL image.
        if 'image_path' in tool.input_style:
            # when the tool is initialized, the description is already
            # formatted with the input description and output description.
            # so we need to update the description after we change the
            # input_style and output_style.
            old_input_description = self.tool.generate_input_description()
            self.tool.input_style = self.tool.input_style.replace(
                'image_path', 'pil image')
            new_input_description = self.tool.generate_input_description()
            old_description = self.tool.toolmeta.description
            self.tool.toolmeta.description = old_description.replace(
                old_input_description, new_input_description)

        if 'image_path' in tool.output_style:
            old_output_description = self.tool.generate_output_description()
            self.tool.output_style = self.tool.output_style.replace(
                'image_path', 'pil image')
            new_output_description = self.tool.generate_output_description()
            old_description = self.tool.toolmeta.description
            self.tool.toolmeta.description = old_description.replace(
                old_output_description, new_output_description)

        self.description = tool.toolmeta.description

    def __call__(self, *args, **kwargs):
        return self.tool(*args, **kwargs)


def load_mmtools_for_tf_agent(load_list=[], device='cpu'):
    """Load mmtools into transformers agent style.

    Args:
        load_list (list): list of mmtools
        device (str): device to load mmtools

    Returns:
       tools (list): list of mmtools
    """
    tools = []
    tool_list = load_list if len(load_list) else list_tools()
    for tool_name in tool_list:
        mmtool = load_tool(tool_name, device=device)
        hf_tool = ToolAdapter(mmtool)
        tools.append(hf_tool)

    return tools
