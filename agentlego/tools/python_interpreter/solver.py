from func_timeout import func_set_timeout

from agentlego.types import Annotated, Info
from agentlego.utils import require
from ..base import BaseTool
from .python_interpreter import GenericRuntime

DESC_EN = '''\
This tool can execute Python code to solve math equations. The code should include a function named 'solution'. You should use the `sympy` library in your code to solve the equations. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
from sympy import symbols, Eq, solve
def solution():
    # Define symbols
    x, y = symbols('x y')

    # Define equations
    equation1 = Eq(x**2 + y**2, 20)
    equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)

    # Solve the system of equations
    solutions = solve((equation1, equation2), (x, y), dict=True)

    # Return solutions as strings
    return str(solutions)
```
'''  # noqa: E501


class Solver(BaseTool):
    """A tool to solve math equations using python interpreter.

    Args:
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = DESC_EN
    answer_expr = 'solution()'

    @require('sympy')
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

    def _call(self, command: str) -> str:
        runtime = GenericRuntime(headers=['from sympy import symbols, Eq, solve'])
        runtime.exec_code('\n'.join(command))
        return runtime.eval_code(self.answer_expr)
