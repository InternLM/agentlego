import re
from typing import Optional, Tuple


def parse_multi_float(
    input_str: str,
    number: Optional[int] = None,
) -> Tuple[float, ...]:
    pattern = r'([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, input_str)

    if number is not None and len(matches) != number:
        raise ValueError(f'Expected {number} numbers, got {input_str}.')
    else:
        return tuple(float(num) for num in matches)
