# Test coverage report

This document tracks unit-test coverage of the **Python** portion of `pedalboard`
(everything under `pedalboard/*.py`). The C++ bindings in `pedalboard_native` are
compiled code and are out of scope for line coverage; they are exercised indirectly
by the existing test-suite and were **not modified** in any way.

## How the numbers were produced

```bash
python -m venv .venv && . .venv/bin/activate
pip install pedalboard==0.9.24 pytest pytest-cov pytest-mock mido mutagen psutil tqdm

# Run from the repository root so that `import pedalboard` resolves to the source
# tree (pedalboard/*.py) while the compiled `pedalboard_native` module comes from
# the pre-built v0.9.24 wheel (identical version to the checked-out sources).
python -m pytest tests --cov=pedalboard --cov-report=term-missing \
    --ignore=tests/test_benchmark.py \
    --ignore=tests/test_memory_leaks.py \
    --ignore=tests/test_tensorflow.py \
    --ignore=tests/test_type_hints.py \
    --ignore=tests/test_external_plugins.py
```

Excluded test modules (identically for *before* and *after*, so the comparison is
apples-to-apples):

| Module | Reason |
| --- | --- |
| `test_external_plugins.py` | Requires downloading third-party VST3 binaries (`tests/download_test_plugins.py`); not available in the CI sandbox. |
| `test_benchmark.py`, `test_memory_leaks.py` | Performance/leak checks, not unit tests. |
| `test_tensorflow.py` | Requires TensorFlow. |
| `test_type_hints.py` | Runs `mypy`/`pyright` over stubs, not runtime coverage. |

Environment: Linux x86_64, Python 3.10.12, `pedalboard` 0.9.24, `pytest-cov`.

## Baseline (before) — 26,491 tests passed, 5 skipped

| File | Stmts | Miss | Cover |
| --- | ---: | ---: | ---: |
| `pedalboard/__init__.py` | 12 | 4 | 67% |
| `pedalboard/_pedalboard.py` | 405 | 316 | **22%** |
| `pedalboard/io/__init__.py` | 2 | 0 | 100% |
| `pedalboard/midi_utils.py` | 38 | 26 | **32%** |
| `pedalboard/version.py` | 2 | 0 | 100% |
| **TOTAL** | **459** | **346** | **25%** |

### Why these two files are the highest-risk

* **`pedalboard/_pedalboard.py` (22%)** is the largest Python file in the package and
  contains all of the *parameter type-inference* logic behind `plugin.parameters` and
  `plugin.<param> = value` for external VST3/AU plugins: `AudioProcessorParameter`,
  `strip_common_float_suffixes`, `looks_like_float`, `normalize_python_parameter_name`,
  `wrap_type`/`WrappedBool`, `_PythonExternalPluginMixin` and `load_plugin`. Almost all
  of its coverage previously came from `test_external_plugins.py`, which depends on
  downloading real third-party plugin binaries — so in any environment without those
  binaries this logic was effectively untested.
* **`pedalboard/midi_utils.py` (32%)** normalizes user-supplied MIDI messages before
  they are handed to C++. The only existing test covered the happy path with `mido`
  messages; all error handling and the "delta vs. absolute timestamps" heuristic were
  uncovered.

## After — 26,634 tests passed, 5 skipped (+143 new tests)

| File | Stmts | Miss | Cover | Δ |
| --- | ---: | ---: | ---: | ---: |
| `pedalboard/__init__.py` | 12 | 4 | 67% | — |
| `pedalboard/_pedalboard.py` | 405 | 15 | **96%** | +74 pp |
| `pedalboard/io/__init__.py` | 2 | 0 | 100% | — |
| `pedalboard/midi_utils.py` | 38 | 0 | **100%** | +68 pp |
| `pedalboard/version.py` | 2 | 0 | 100% | — |
| **TOTAL** | **459** | **19** | **96%** | **+71 pp** |

### New test files

| File | Tests | What it covers |
| --- | ---: | --- |
| `tests/test_parameter_helpers.py` | 74 | Pure helpers in `_pedalboard.py`: suffix stripping (`dB`, `Hz`, `kHz`, `%`, …), `looks_like_float`, Python-name normalization (`C#` → `c_sharp`), `ReadOnlyDictWrapper`, `WrappedBool`, the `wrap_type` weak-reference wrappers, `Pedalboard.__repr__`. |
| `tests/test_audio_processor_parameter.py` | 45 | `AudioProcessorParameter` type inference (float / bool / string), range & step detection, label inference, `get_raw_value_for` validation and fallback paths, the slow-fetch fallback, parameter disappearance, `_PythonExternalPluginMixin` (`.parameters`, `__getattr__`/`__setattr__`, `__dir__`, initial parameter values, ignored `MIDI CC`/`Pxxx` parameters) and `load_plugin` error paths. Uses a small in-Python fake of the C++ `_AudioProcessorParameter`, so **no plugin binaries are needed**. |
| `tests/test_midi_utils_normalization.py` | 24 | `parse_midi_message_part` for every input type, `parse_midi_message_string`, all `normalize_midi_messages` branches including the delta-timestamp heuristic thresholds. |

### Remaining uncovered lines

* `pedalboard/__init__.py` 26–35: the Windows-only "DLL load failed" import error hint.
* `pedalboard/_pedalboard.py`: a handful of defensive branches that are unreachable
  with well-behaved inputs (`super().__getattr__` fallbacks, `raise ValueError` for an
  "impossible" parameter type, macOS-only `AudioUnitPlugin` import).

No library source code (`.py`, `.cpp`, `.h`) was changed; the only additions are the
three test files above, this document, and `NOTICE.md`.
