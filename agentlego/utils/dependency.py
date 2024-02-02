import re
import warnings
from functools import wraps
from importlib.metadata import PackageNotFoundError, distribution
from inspect import isfunction

from packaging.version import parse


def _digit_version(version_str: str, length: int = 4):
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


def _check_dependency(dep):
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
        if op is None or getattr(_digit_version(dist.version), op)(
                _digit_version(version)):
            return True
    except PackageNotFoundError:
        pass

    return False


PACKAGE_AVAILABILITY = dict()


def is_package_available(dep):
    """Check if the package is available.

    Args:
        dep (str): The dependency package name,
            like ``transformers`` or ``transformers>=4.28.0``.

    Returns:
        bool: True if the package is available, False otherwise.
    """

    if dep not in PACKAGE_AVAILABILITY:
        PACKAGE_AVAILABILITY[dep] = _check_dependency(dep)
    return PACKAGE_AVAILABILITY[dep]


def require(dep, install=None):
    """A wrapper of function for extra package requirements.

    Args:
        dep (Sequence[str) | str): The dependency package name,
            like ``transformers`` or ``transformers>=4.28.0``.
        install (str, optional): The installation command hint. Defaults
            to None, which means to use "pip install dep".
    """
    if isinstance(dep, str):
        dep = [dep]

    def wrapper(fn):
        assert isfunction(fn)

        @wraps(fn)
        def ask_install(*args, **kwargs):
            msg = '{name} requires {dep}, please install by `{ins}`.'.format(
                name=fn.__qualname__.replace('.__init__', ''),
                dep=', '.join(dep),
                ins=install or 'pip install {}'.format(' '.join(repr(i) for i in dep)))
            raise ImportError(msg)

        if all(_check_dependency(item) for item in dep):
            fn._verify_require = getattr(fn, '_verify_require', lambda: None)
            return fn

        ask_install._verify_require = ask_install
        return ask_install

    return wrapper
