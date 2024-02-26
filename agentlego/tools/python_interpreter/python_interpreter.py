import copy
from typing import Any, Optional

from func_timeout import func_set_timeout

from agentlego.types import Annotated, Info
from ..base import BaseTool

DESC_EN = '''\
This tool can execute Python code. The code should include a function named 'solution'. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
import xxx
def solution():
    # python code to get the final answer
    ...
    return final_answer
```
'''  # noqa: E501


class GenericRuntime:

    def __init__(
        self,
        global_dict: Optional[dict] = None,
        local_dict: Optional[dict] = None,
        headers: list = [],
    ):
        self._global_vars = copy.copy(global_dict) if global_dict else {}
        self._local_vars = copy.copy(local_dict) if local_dict else {}

        for c in headers:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars, self._local_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars, self._local_vars)


class PythonInterpreter(BaseTool):
    """A Python executor that can execute Python scripts.

    WARNING: The PythonInterpreter only has minimal protection, don't expose to
    trustless environment.

    Args:
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = DESC_EN
    answer_expr = 'solution()'

    def __init__(self, timeout: int = 20, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.timeout = timeout

    def apply(self, command: Annotated[str, Info('Markdown format Python code')]) -> str:

        if '```python' in command:
            command = command.split('```python')[1].split('```')[0]
        elif '```' in command:
            command = command.split('```')[1].split('```')[0]

        res = func_set_timeout(self.timeout)(self._call)(command)
        return str(res)

    def _call(self, command: str) -> Any:
        runtime = GenericRuntime()
        runtime.exec_code('\n'.join(command))
        return runtime.eval_code(self.answer_expr)
