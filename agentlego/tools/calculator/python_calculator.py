import math
from typing import Callable, Union

import addict
from func_timeout import func_timeout

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from ..base import BaseTool


def safe_eval(expr):
    math_methods = {
        k: v
        for k, v in math.__dict__.items() if not k.startswith('_')
    }
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
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Calculator',
        description='A calculator tool. The input must be a single Python '
        'expression and you cannot import packages. You can use functions '
        'in the `math` package without import.',
        inputs=['text'],
        outputs=['text'],
    )

    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 timeout=2):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.timeout = timeout

    def apply(self, expression: str) -> str:
        res = func_timeout(self.timeout, safe_eval, [expression])
        return str(res)
