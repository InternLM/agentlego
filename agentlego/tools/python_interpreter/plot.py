import copy
import io
from typing import Any, Optional, Type
from ..base import BaseTool
import os
from agentlego.utils import temp_path


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


class Plot(BaseTool):
    """A tool to plot diagrams using python interpreter.

    Args:
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = (
        f""" This tool can execute Python code to plot diagrams. The code must be a function, the function name must be 'solution'. The parameter must be 'path'. You should use Matplotlib library in your code to plot diagrams. In the end of the code, you must save diagrams using 'plt.savefit(path)' and return the path 'return path'. The code corresponds to your thinking process. A code instance example is as follows:

        ```python
        # import packages
        import matplotlib.pyplot as plt
        def solution(path):
            # labels and data
            cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
            data = [23, 17, 35, 29, 12, 41]

            # draw diagrams
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title('Car Distribution')

            # save diagrams
            plt.savefig(path)
            return path
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

            self.runtime.exec_code("\n".join(command))
            path = temp_path('image', '.png')
            expr = f"{self.answer_expr.split('(')[0]}(\'{path}\')"
            res = self.runtime.eval_code(expr)
        except Exception as e:
            print(f'{type(e)}: {str(e)}')
            return repr(e)
        try:
            res = str(res)
        except Exception as e:
            print(f'{type(e)}: {str(e)}')
            return repr(e)
        return res
