# Tool Server

AgentLego provides a suit of tool server utilities to help you deploy tools on a server and use it like local
tools on clients.

## Start a server

We provide a command-line tool `agentlego-server` to start a tool server. You can specify the tool names you want to use.

```bash
agentlego-server start Calculator ImageDescription TextToImage
```

And then, the server will setup all tools and start.

```bash
INFO:     Started server process [1741344]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:16180 (Press CTRL+C to quit)
```

## Use tools in client

In the client, you can create a remote tool from the url of the tool server.

```python
from agentlego.tools.remote import RemoteTool

# Create all remote tools from a tool server root url.
tools = RemoteTool.from_server('http://127.0.0.1:16180')
for tool in tools:
    print(tool.name, tool.url)

# Create single remote tool from a tool server endpoint.
# All endpoint can be found in the docs of the tool server, like http://127.0.0.1:16180/docs
tool = RemoteTool.from_url('http://127.0.0.1:16180/ImageDescription')
print(tool.description)
```

## How to Deploy Your Own Tools

`agentlego-server` accepts additional tool modules, which means you don't need to modify the source code of `AgentLego`. You just need to write your tool source code in a Python file or module to deploy tools using `agentlego-server`.

First, we create a Python file named `my_tool.py`

```python
from agentlego.tools import BaseTool

class Clock(BaseTool):
    default_desc = 'Returns the current date and time.'

    def apply(self) -> str:
        from datetime import datetime
        return datetime.now().strftime('%Y/%m/%d %H:%M')

class RandomNumber(BaseTool):
    default_desc = 'Returns a random number not greater than `max`'

    def apply(self, max: int) -> int:
        import random
        return random.randint(0, max)
```

In this file, we defined two tools: `Clock` and `RandomNumber`. After saving the file, use the following command in the command line to check if `agentlego-server` can correctly read these two tools:

```bash
# We use the --extra option to specify the additional tool source file
# Use the --no-official option to hide the built-in tools of AgentLego
agentlego-server list --extra ./my_tool.py --no-official
```

Getting the following output means that `agentlego-server` can read these tools

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Class        ┃ source           ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Clock        │ /home/my_tool.py │
│ RandomNumber │ /home/my_tool.py │
└──────────────┴──────────────────┘
```

Start the tool server:

```bash
agentlego-server start --extra ./my_tool.py Clock RandomNumber
```
