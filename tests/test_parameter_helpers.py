#! /usr/bin/env python
#
# Copyright 2021 Spotify AB
#
# Licensed under the GNU Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/gpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false

"""
Unit tests for the pure-Python helpers in ``pedalboard/_pedalboard.py``
that do not require loading an external (VST3/AU) plugin.
"""

import pytest

import pedalboard
from pedalboard import Pedalboard
from pedalboard._pedalboard import (
    FLOAT_SUFFIXES_TO_IGNORE,
    BooleanWithParameter,
    FloatWithParameter,
    ReadOnlyDictWrapper,
    StringWithParameter,
    WrappedBool,
    _SuffixTrie,
    looks_like_float,
    normalize_python_parameter_name,
    strip_common_float_suffixes,
    to_python_parameter_name,
    wrap_type,
)


class TestStripCommonFloatSuffixes:
    @pytest.mark.parametrize("value", [1.0, 0, True, None, [1, 2]])
    def test_non_strings_are_returned_untouched(self, value):
        assert strip_common_float_suffixes(value) is value

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.5 dB", "1.5"),
            ("1.5dB", "1.5"),
            ("  440 Hz  ", "440"),
            ("440 HZ", "440"),
            ("50%", "50"),
            ("2x", "2"),
            ("10 ms", "10"),
            ("3 sec", "3"),
            ("3 seconds", "3"),
            ("-6.0 dBTP", "-6.0"),
            ("1,", "1"),
            ("1.", "1"),
            ("1*", "1"),
        ],
    )
    def test_strips_single_suffix(self, raw, expected):
        assert strip_common_float_suffixes(raw) == expected

    def test_strips_multiple_suffixes_repeatedly(self):
        # "ms" is stripped, then the trailing "." is stripped as well.
        assert strip_common_float_suffixes("10. ms") == "10"

    @pytest.mark.parametrize("raw", ["hello", "1 foo", "", "dBx1"])
    def test_leaves_strings_without_known_suffixes_alone(self, raw):
        assert strip_common_float_suffixes(raw) == raw.strip()

    def test_khz_is_converted_to_hz(self):
        assert strip_common_float_suffixes("1.5 kHz") == "1500.0"
        assert strip_common_float_suffixes("2kHz") == "2000.0"

    def test_khz_without_a_number_is_returned_unchanged(self):
        assert strip_common_float_suffixes("abc kHz") == "abc kHz"

    def test_khz_is_not_converted_when_si_prefixes_are_disabled(self):
        # With strip_si_prefixes=False, "khz" is not a known suffix; only "hz" is,
        # so the trailing "hz" is removed and the "k" remains.
        assert (
            strip_common_float_suffixes("1.5 kHz", strip_si_prefixes=False) == "1.5 k"
        )

    def test_all_known_suffixes_are_in_the_trie(self):
        for suffix in FLOAT_SUFFIXES_TO_IGNORE:
            assert strip_common_float_suffixes(f"42 {suffix}") == "42"


class TestSuffixTrie:
    def test_repr_is_informative(self):
        node = _SuffixTrie("z")
        node.is_match = True
        assert "_SuffixTrie" in repr(node)
        assert "z" in repr(node)
        assert "True" in repr(node)


class TestLooksLikeFloat:
    @pytest.mark.parametrize(
        "value", [1.0, -3.5, "1", "1.5 dB", "  440 Hz", "1e3", "2 kHz"]
    )
    def test_true_cases(self, value):
        assert looks_like_float(value) is True

    @pytest.mark.parametrize("value", ["hello", "", "1 foo", "on", "off"])
    def test_false_cases(self, value):
        assert looks_like_float(value) is False


class TestNormalizePythonParameterName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Mix", "mix"),
            ("Delay Time (ms)", "delay_time_ms"),
            ("  Wet/Dry  ", "wet_dry"),
            ("A__B", "a_b"),
            ("___leading and trailing___", "leading_and_trailing"),
            ("C#", "c_sharp"),
            ("C♯", "c_sharp"),
            ("B♭", "b_flat"),
            ("Ünïcödé", "n_c_d"),
            ("Level 2", "level_2"),
            ("", ""),
            ("___", ""),
        ],
    )
    def test_normalization(self, raw, expected):
        assert normalize_python_parameter_name(raw) == expected


class _FakeCppParameter:
    def __init__(self, name: str, label: str = ""):
        self.name = name
        self.label = label


class TestToPythonParameterName:
    def test_returns_none_when_name_and_label_are_empty(self):
        assert to_python_parameter_name(_FakeCppParameter("", "")) is None

    def test_name_only(self):
        assert to_python_parameter_name(_FakeCppParameter("Cutoff")) == "cutoff"

    def test_label_is_appended(self):
        assert (
            to_python_parameter_name(_FakeCppParameter("Cutoff", "Hz")) == "cutoff_hz"
        )

    def test_label_starting_with_colon_is_ignored(self):
        assert to_python_parameter_name(_FakeCppParameter("Cutoff", ":Hz")) == "cutoff"


class TestReadOnlyDictWrapper:
    def test_reads_like_a_dict(self):
        wrapped = ReadOnlyDictWrapper({"a": 1})
        assert wrapped["a"] == 1
        assert list(wrapped.keys()) == ["a"]

    def test_assignment_raises_a_helpful_error(self):
        wrapped = ReadOnlyDictWrapper({"a": 1})
        with pytest.raises(TypeError, match=r"my_plugin\.a = 2"):
            wrapped["a"] = 2


class TestWrappedBool:
    def test_requires_a_bool(self):
        with pytest.raises(TypeError, match="should be passed a boolean"):
            WrappedBool(1)

    @pytest.mark.parametrize("value", [True, False])
    def test_behaves_like_the_wrapped_bool(self, value):
        wrapped = WrappedBool(value)
        assert bool(wrapped) is value
        assert (wrapped == value) is True
        assert (wrapped == (not value)) is False
        assert hash(wrapped) == hash(value)
        assert repr(wrapped) == repr(value)
        assert str(wrapped) == str(value)

    def test_delegates_attribute_access_to_the_bool(self):
        assert WrappedBool(True).bit_length() == 1
        assert WrappedBool(False).bit_length() == 0

    def test_usable_in_boolean_contexts(self):
        assert WrappedBool(True) and not WrappedBool(False)


class _Wrapped:
    """A stand-in for an AudioProcessorParameter."""

    def __init__(self):
        self.label = "dB"
        self.extra = "value"


class TestWrapType:
    def test_requires_wrapped_kwarg(self):
        with pytest.raises(ValueError, match="expected to be passed a 'wrapped' kwarg"):
            FloatWithParameter(1.0)

    def test_float_wrapper_behaves_like_a_float_and_exposes_wrapped_attrs(self):
        wrapped = _Wrapped()
        value = FloatWithParameter(1.5, wrapped=wrapped)
        assert isinstance(value, float)
        assert value == 1.5
        assert value + 1 == 2.5
        assert value.label == "dB"
        assert value.extra == "value"
        assert "label" in dir(value)
        assert "real" in dir(value)

    def test_string_wrapper_behaves_like_a_string(self):
        wrapped = _Wrapped()
        value = StringWithParameter("hello", wrapped=wrapped)
        assert isinstance(value, str)
        assert value == "hello"
        assert value.upper() == "HELLO"
        assert value.label == "dB"

    def test_boolean_wrapper_behaves_like_a_bool(self):
        wrapped = _Wrapped()
        value = BooleanWithParameter(True, wrapped=wrapped)
        assert bool(value) is True
        assert value == True  # noqa: E712
        assert value.label == "dB"

    def test_missing_attribute_raises_attribute_error(self):
        wrapped = _Wrapped()
        value = FloatWithParameter(1.5, wrapped=wrapped)
        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            value.does_not_exist

    def test_dir_falls_back_when_wrapped_object_is_gone(self):
        wrapped = _Wrapped()
        value = FloatWithParameter(1.5, wrapped=wrapped)
        del wrapped
        assert "label" not in dir(value)
        assert "real" in dir(value)

    def test_wrap_type_creates_a_subclass(self):
        Wrapper = wrap_type(int)
        assert issubclass(Wrapper, int)
        assert Wrapper(3, wrapped=_Wrapped()) == 3


class TestPedalboardRepr:
    def test_empty(self):
        assert repr(Pedalboard()) == "<Pedalboard with 0 plugins: []>"

    def test_single_plugin_uses_singular_noun(self):
        board = Pedalboard([pedalboard.Gain(gain_db=0)])
        text = repr(board)
        assert text.startswith("<Pedalboard with 1 plugin: [")
        assert "Gain" in text

    def test_multiple_plugins_use_plural_noun(self):
        board = Pedalboard([pedalboard.Gain(gain_db=0), pedalboard.Gain(gain_db=0)])
        assert repr(board).startswith("<Pedalboard with 2 plugins: [")

    def test_default_constructor_is_empty_list(self):
        assert list(Pedalboard()) == []
        assert list(Pedalboard(None)) == []
