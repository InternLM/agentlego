# Copyright (c) OpenMMLab. All rights reserved.
import re
import warnings
from functools import wraps
from inspect import isfunction

from importlib_metadata import PackageNotFoundError, distribution
from packaging.version import parse


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


def satisfy_requirement(dep):
    pat = '(' + '|'.join(['>=', '==', '>']) + ')'
    parts = re.split(pat, dep, maxsplit=1)
    parts = [p.strip() for p in parts]
    package = parts[0]
    if len(parts) > 1:
        op, version = parts[1:]
        op = {
            '>=': '__ge__',
            '==': '__eq__',
            '>': '__gt__',
            '<': '__lt__',
            '<=': '__le__'
        }[op]
    else:
        op, version = None, None

    try:
        dist = distribution(package)
        if op is None or getattr(digit_version(dist.version), op)(
                digit_version(version)):
            return True
    except PackageNotFoundError:
        pass

    return False


def require(dep, install=None):
    """A wrapper of function for extra package requirements.

    Args:
        dep (str): The dependency package name, like ``transformers``
            or ``transformers>=4.28.0``.
        install (str, optional): The installation command hint. Defaults
            to None, which means to use "pip install dep".
    """

    def wrapper(fn):
        assert isfunction(fn)

        @wraps(fn)
        def ask_install(*args, **kwargs):
            name = fn.__qualname__.replace('.__init__', '')
            ins = install or f'pip install "{dep}"'
            raise ImportError(
                f'{name} requires {dep}, please install it by `{ins}`.')

        if satisfy_requirement(dep):
            fn._verify_require = getattr(fn, '_verify_require', lambda: None)
            return fn

        ask_install._verify_require = ask_install
        return ask_install

    return wrapper
