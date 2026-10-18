"""Zero-dependency test runner.

The tests are written to be pytest-compatible, but this runner lets them run
with only torch + numpy installed:

    PYTHONPATH=src python tests/run_tests.py
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import traceback


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    here = pathlib.Path(__file__).parent
    sys.path.insert(0, str(here.parent / "src"))
    passed = failed = 0
    for path in sorted(here.glob("test_*.py")):
        mod = _load(path)
        for name, fn in sorted(vars(mod).items()):
            if name.startswith("test_") and callable(fn) and not inspect.isfunction(fn):
                continue
            if name.startswith("test_") and inspect.isfunction(fn):
                try:
                    fn()
                    passed += 1
                    print(f"PASS {path.name}::{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL {path.name}::{name}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
