# 工具服务器

AgentLego 提供了一套工具服务器辅助程序，帮助您在服务器上部署工具，并在客户端上像使用本地工具一样调用这些工具。

## 启动服务器

我们提供了一个命令行工具 `agentlego-server` 来启动工具服务器。您可以指定要启动的工具类别。

```bash
agentlego-server start Calculator ImageDescription TextToImage
```

然后，服务器将启动所有工具。

```bash
INFO:    Started server process [1741344]
INFO:    Waiting for application startup.
INFO:    Application startup complete.
INFO:    Uvicorn running on http://127.0.0.1:16180 (Press CTRL+C to quit)
```

## 在客户端使用工具

在客户端，您可以使用工具服务器的 URL 创建所有远程工具。

```python
from agentlego.tools.remote import RemoteTool

# 从工具服务器 URL 创建所有远程工具。
tools = RemoteTool.from_server('http://127.0.0.1:16180')
for tool in tools:
   print(tool.name, tool.url)

# 从工具服务器端点创建单个远程工具。
# 所有端点都可以在工具服务器的文档中找到，例如 http://127.0.0.1:16180/docs
tool = RemoteTool.from_url('http://127.0.0.1:16180/ImageDescription')
print(tool.description)
```

## 如何部署自己的工具

`agentlego-server` 接受额外的工具模块，这意味着你不需要修改 `AgentLego` 的源码，只需要在一个 Python 文件或者模
块里编写你的工具源码，即可使用 `agentlego-server` 部署工具。

首先，我们新建一个 Python 文件，名称为 `my_tool.py`

```python
from agentlego.tools import BaseTool

class Clock(BaseTool):
    default_desc = '返回当前日期和时间的时钟。'

    def apply(self) -> str:
        from datetime import datetime
        return datetime.now().strftime('%Y/%m/%d %H:%M')

class RandomNumber(BaseTool):
    default_desc = '返回一个不大于 `max` 的随机数'

    def apply(self, max: int) -> int:
        import random
        return random.randint(0, max)
```

在这个文件中，我们定义了两个工具 `Clock` 和 `RandomNumber`，保存文件之后，在命令行中，使用如下命令，检查
`agentlego-server` 是否能够正确读取这两个工具：

```bash
# 我们使用 --extra 选项，指定额外的工具源码文件
# 使用 --no-official 选项隐藏 AgentLego 内置的工具
agentlego-server list --extra ./my_tool.py --no-official
```

获得如下输出，说明 `agentlego-server` 能够读取这些工具

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Class        ┃ source           ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Clock        │ /home/my_tool.py │
│ RandomNumber │ /home/my_tool.py │
└──────────────┴──────────────────┘
```

启动工具服务器：

```bash
agentlego-server start --extra ./my_tool.py Clock RandomNumber
```
