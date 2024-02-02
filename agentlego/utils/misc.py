from typing import Any, Callable


def apply_to(data: Any, expr: Callable, apply_func: Callable):
    """Apply function to each element in dict, list or tuple that matches with
    the expression.

    For examples, if you want to convert each element in a list of dict from
    `np.ndarray` to `Tensor`. You can use the following code:

    Examples:
        >>> from agentlego.utils import apply_to
        >>> import numpy as np
        >>> import torch
        >>> data = dict(array=[np.array(1)]) # {'array': [array(1)]}
        >>> result = apply_to(data, lambda x: isinstance(x, np.ndarray), lambda x: torch.from_numpy(x))
        >>> print(result) # {'array': [tensor(1)]}

    Args:
        data (Any): Data to be applied.
        expr (Callable): Expression to tell which data should be applied with
            the function. It should return a boolean.
        apply_func (Callable): Function applied to data.

    Returns:
        Any: The data after applying.
    """  # noqa: E501
    if isinstance(data, dict):
        # Keep the original dict type
        res = type(data)()
        for key, value in data.items():
            res[key] = apply_to(value, expr, apply_func)
        return res
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        # namedtuple
        return type(data)(*(apply_to(sample, expr, apply_func) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, (tuple, list)):
        return type(data)(apply_to(sample, expr, apply_func) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
    elif expr(data):
        return apply_func(data)
    else:
        return data
