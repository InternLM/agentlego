import copy
import io
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


class PythonInterpreter(BaseTool):
    """A Python executor that can execute Python scripts.

    Args:
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = (
        """ This tool can execute Python code. The code must be a function, the function name must be 'solution', and the code corresponds to your thinking process. The solution function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

        ```python
        # import packages
        import xxx
        def solution():
            # initialize variables
            variable_names_with_real_meaning = xxx
            # step 1
            mid_variable = func(variable_names_with_real_meaning)
            # step x
            mid_variable = func(mid_variable)
            # result
            final_answer = func(mid_variable)
            return final_answer
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
