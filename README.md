# OpenMMLab Visual Toolbox for LLMs

## Visual ChatGPT

```Python
from mmlmtools import list_tool, load_tool

self.tools = []
self.models = {}

mmtools = list_tool()  # get the list of mmtools

for tool_name in mmtools:
    # obtain tool instance and toolmeta via `load_tool()`
    mmtool, toolmeta = load_tool(tool_name, device='cpu')  

    self.models[tool_name] = mmtool
    self.tools.append(
        Tool(
            name=toolmeta.tool_name,
            description=toolmeta.description,
            func=mmtool.apply))
```