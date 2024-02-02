import importlib
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Union


def resolve_module(name_or_path: Union[Path, str]) -> ModuleType:
    if isinstance(name_or_path, str):
        try:
            module = importlib.import_module(name_or_path)
            return module
        except Exception:
            name_or_path = Path(name_or_path)

    name = '_ext_' + name_or_path.stem.replace(' ', '').replace('-', '_')
    if name_or_path.is_dir():
        name_or_path = name_or_path / '__init__.py'

    if not name_or_path.exists():
        raise ImportError(f'Cannot import from `{name_or_path}` '
                          'since the path does not exist.')

    spec = spec_from_file_location(name, str(name_or_path))
    if spec is None:
        raise ImportError(f'Failed to import from `{name_or_path}`.')
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    del sys.modules[name]
    return module
