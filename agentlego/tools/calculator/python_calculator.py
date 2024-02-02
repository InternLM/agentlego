import math

import addict
from func_timeout import func_timeout

from ..base import BaseTool


def safe_eval(expr):
    math_methods = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
    allowed_methods = {
        'math': addict.Addict(math_methods),
        'max': max,
        'min': min,
        'round': round,
        'sum': sum,
        **math_methods,
    }
    allowed_methods['__builtins__'] = None
    return eval(expr, allowed_methods, allowed_methods)


class Calculator(BaseTool):
    """A calculator based on Python expression.

    Args:
        timeout (int): The timeout value to interrupt calculation.
            Defaults to 2.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('A calculator tool. The input must be a single Python '
                    'expression and you cannot import packages. You can use functions '
                    'in the `math` package without import.')

    def __init__(self, timeout=2, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.timeout = timeout

    def apply(self, expression: str) -> str:
        res = func_timeout(self.timeout, safe_eval, [expression])
        return str(res)
