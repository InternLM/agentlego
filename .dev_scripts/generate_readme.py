import argparse
import inspect

from agentlego.apis.tool import NAMES2TOOLS

prog_description = """\
A tool to generate a README template for a tool.
"""

README_TMPL = '''\
# {cls_name}

## Default Tool Meta

- **name**: {name}
- **description**: {description}
- **inputs**: {inputs}
- **outputs**: {outputs}

## Examples
{download}

**Use the tool directly (without agent)**

```python
from agentlego.apis import load_tool

# load tool
tool = load_tool('{cls_name}'{init_args})

# apply tool
TODO
```

**With Lagent**

```python
from lagent import ReAct, GPTAPI, ActionExecutor
from agentlego.apis.agents import load_tools_for_lagent

# load tools and build agent
# please set `OPENAI_API_KEY` in your environment variable.
tools = load_tools_for_lagent(tools=['{cls_name}']{init_args})
agent = ReAct(GPTAPI(temperature=0.), action_executor=ActionExecutor(tools))

# agent running with the tool.
ret = agent.chat(f'TODO')
for step in ret.inner_steps[1:]:
    print('------')
    print(step['content'])
```

## Set up

Before using the tool, please confirm you have installed the related dependencies by the below commands.

```bash
TODO
```

## Reference

TODO
'''  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'tools', type=str, nargs='+', help='The tool class to generate.')
    args = parser.parse_args()
    return args


def generate_readme(tool: type):
    toolmeta = tool.DEFAULT_TOOLMETA
    if set(toolmeta.inputs) & {'image', 'audio', 'video'}:
        download = '''
**Download the demo resource**

```bash
TODO
```'''
    else:
        download = ''

    if 'device' in inspect.getfullargspec(tool).args:
        init_args = ", device='cuda'"
    else:
        init_args = ''

    readme = README_TMPL.format(
        cls_name=tool.__name__,
        name=toolmeta.name,
        description=toolmeta.description,
        inputs=', '.join(toolmeta.inputs),
        outputs=', '.join(toolmeta.outputs),
        download=download,
        init_args=init_args,
    )
    return readme


def main():
    args = parse_args()
    for tool in args.tools:
        assert tool in NAMES2TOOLS, f'Not found `{tool}`.'
        print(generate_readme(NAMES2TOOLS.get(tool)))


if __name__ == '__main__':
    main()
