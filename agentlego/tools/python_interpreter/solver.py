import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type
from ..base import BaseTool


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


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

    default_desc = (
        """ This tool can execute Python code to solve math equations. The code must be a function, the function name must be 'solution', and the code corresponds to your thinking process. You should use SymPy library in your code to solve the given math equations. The solution function should return its answer in the string format. Avoid printing the answer. A code instance example is as follows:

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
            solutions = solve((equation1, equation2), (x, y))

            # Convert solutions to strings
            solutions_str = [f"x = {sol[0]}, y = {sol[1]}" for sol in solutions]

            # Return solutions as strings
            return solutions_str
        ```

        Args:
            command (:class:`str`): Python code snippet
        """
    )

    def __init__(self, 
                 answer_expr: Optional[str] = 'solution()',
                 timeout: int = 20,
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.answer_expr = answer_expr
        self.timeout = timeout

    def apply(self, command: str) -> str:
        from func_timeout import FunctionTimedOut, func_set_timeout
        self.runtime = GenericRuntime()
        try:
            res = func_set_timeout(self.timeout)(self._call)(command)
        except FunctionTimedOut as e:
            print(f'{type(e)}: {str(e)}')
            return repr(e)  # ?
        return res
    
    def _call(self, command: str) -> str:
        try:
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]
            command = command.split('\n')

            self.runtime.exec_code('\n'.join(command))
            res = self.runtime.eval_code(self.answer_expr)
        except Exception as e:
            print(f'{type(e)}: {str(e)}')
            return repr(e)
        try:
            res = str(res)
        except Exception as e:
            print(f'{type(e)}: {str(e)}')
            return repr(e)
        return res
