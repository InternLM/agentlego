import importlib
import sys
from pathlib import Path
from typing import Union


def resolve_module(module: Union[Path, str]):
    if isinstance(module, str):
        try:
            module = importlib.import_module(module)
            return module
        except Exception:
            module = Path(module)

    name = '_ext_' + module.stem.replace(' ', '').replace('-', '_')
    if module.is_dir():
        module = module / '__init__.py'

    if not module.exists():
        raise ImportError(f'Cannot import from `{module}` '
                          'since the path does not exist.')

    spec = importlib.util.spec_from_file_location(name, str(module))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    del sys.modules[name]
    return module
