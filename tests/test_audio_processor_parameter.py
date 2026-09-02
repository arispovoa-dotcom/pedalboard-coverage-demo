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
Unit tests for :class:`pedalboard.AudioProcessorParameter` and the
``_PythonExternalPluginMixin`` that back the ``plugin.parameters`` /
``plugin.<parameter_name>`` interface of external plugins.

These tests use a small fake of the C++ ``_AudioProcessorParameter`` object,
so that the Python type-inference logic can be tested without having to
download or load a real VST3/AU plugin.
"""

from typing import Callable, Dict, List, Optional

import pytest

import pedalboard
from pedalboard import AudioProcessorParameter
from pedalboard._pedalboard import (
    WrappedBool,
    _PythonExternalPluginMixin,
    get_text_for_raw_value,
    load_plugin,
)


class FakeCppParameter:
    """
    A pure-Python stand-in for the pybind11 ``_AudioProcessorParameter``.

    ``text_for`` maps a raw [0, 1] value to the string the plugin would show.
    ``raw_for_text`` optionally overrides how strings are parsed back to raw values;
    by default it does a reverse lookup on ``text_for``.
    """

    def __init__(
        self,
        name: str,
        text_for: Callable[[float], str],
        label: str = "",
        raw_for_text: Optional[Callable[[str], float]] = None,
        slow_only: bool = False,
    ):
        self.name = name
        self.label = label
        self.raw_value = 0.0
        self._text_for = text_for
        self._raw_for_text = raw_for_text
        self._slow_only = slow_only
        self.index = 0

    @property
    def string_value(self) -> str:
        return self._text_for(self.raw_value)

    def get_text_for_raw_value(self, raw_value: float) -> Optional[str]:
        if self._slow_only:
            # Simulate a misbehaving plugin that only answers when raw_value is set.
            return ""
        return self._text_for(raw_value)

    def get_raw_value_for_text(self, text: str) -> float:
        if self._raw_for_text is not None:
            return self._raw_for_text(text)
        candidates = {
            self._text_for(i / 1000): i / 1000 for i in reversed(range(0, 1001))
        }
        if text in candidates:
            return candidates[text]
        # Like a real plugin, try to parse a numeric string (ignoring a unit suffix):
        try:
            numeric = float(text.split(" ")[0])
        except ValueError:
            raise ValueError(f"No raw value for text {text!r}")
        for candidate_text, raw in candidates.items():
            try:
                if float(candidate_text.split(" ")[0]) == numeric:
                    return raw
            except ValueError:
                continue
        raise ValueError(f"No raw value for text {text!r}")

    def __repr__(self) -> str:
        return f'<FakeCppParameter name="{self.name}" raw_value={self.raw_value}>'


class FakePlugin(_PythonExternalPluginMixin):
    """
    A plugin-like object exposing the two members the mixin needs from C++:
    ``_parameters`` and ``_get_parameter(name)``.
    """

    def __init__(self, parameters: List[FakeCppParameter]):
        self.__dict__["_cpp_parameters"] = {p.name: p for p in parameters}

    @property
    def _parameters(self):
        return list(self._cpp_parameters.values())

    def _get_parameter(self, name: str):
        return self._cpp_parameters.get(name)

    def __repr__(self) -> str:
        return "<FakePlugin>"


def stepped_float(minimum: float, maximum: float, steps: int, suffix: str = ""):
    def text_for(raw: float) -> str:
        step = round(raw * steps)
        value = minimum + (maximum - minimum) * step / steps
        return f"{value:g}{suffix}"

    return text_for


def choices(options: List[str]):
    def text_for(raw: float) -> str:
        index = min(int(raw * len(options)), len(options) - 1)
        return options[index]

    return text_for


class TestGetTextForRawValue:
    def test_fast_path_calls_get_text_for_raw_value(self):
        param = FakeCppParameter("x", stepped_float(0, 10, 10))
        assert get_text_for_raw_value(param, 0.5) == "5"
        assert param.raw_value == 0.0

    def test_slow_path_sets_and_restores_raw_value(self):
        param = FakeCppParameter("x", stepped_float(0, 10, 10), slow_only=True)
        param.raw_value = 0.2
        assert get_text_for_raw_value(param, 0.5, slow=True) == "5"
        assert param.raw_value == 0.2


class TestFloatParameter:
    def test_infers_float_range_and_step(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        assert param.type is float
        assert param.min_value == -12
        assert param.max_value == 12
        assert param.step_size == 1
        assert param.approximate_step_size is None
        assert param.range == (-12, 12, 1)
        assert param.python_name == "gain"
        assert set(param.valid_values) == set(range(-12, 13))

    def test_label_is_inferred_from_string_values_when_plugin_has_no_label(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        assert param.label == "dB"
        assert param.units == "dB"

    def test_label_from_plugin_takes_precedence(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(0, 1, 10), label="%")]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        assert param.label == "%"
        assert param.python_name == "gain"

    def test_no_label_when_values_have_no_suffix(self):
        plugin = FakePlugin([FakeCppParameter("Gain", stepped_float(0, 1, 10))])
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        assert param.label is None
        assert param.units is None

    def test_approximate_step_size_for_uneven_steps(self):
        def text_for(raw: float) -> str:
            # Non-linear mapping: values are 0, 1, 4, 9, ... (uneven first derivative)
            return str(round(raw * 4) ** 2)

        plugin = FakePlugin([FakeCppParameter("Curve", text_for)])
        param = AudioProcessorParameter(plugin, "Curve", search_steps=100)
        assert param.type is float
        assert param.step_size is None
        assert param.approximate_step_size == pytest.approx((1 + 3 + 5 + 7) / 4)

    def test_repr_variants(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        assert "range=(-12.0, 12.0, 1.0)>" in repr(param)

        def uneven(raw: float) -> str:
            return str(round(raw * 4) ** 2)

        plugin = FakePlugin([FakeCppParameter("Curve", uneven)])
        param = AudioProcessorParameter(plugin, "Curve", search_steps=100)
        assert "~4.0)>" in repr(param)

    def test_get_raw_value_for_accepts_numbers_and_suffixed_strings(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        # Each value covers a band of raw values (1/24 wide); any raw value in the band is fine.
        assert param.get_raw_value_for(0) == pytest.approx(0.5, abs=1 / 48)
        assert param.get_raw_value_for("6 dB") == pytest.approx(0.75, abs=1 / 48)
        assert param.get_raw_value_for("6dB") == pytest.approx(0.75, abs=1 / 48)
        assert param.get_raw_value_for(-12) == pytest.approx(0.0, abs=1 / 48)
        assert param.get_raw_value_for(12) == pytest.approx(1.0, abs=1 / 48)

    def test_get_raw_value_for_rejects_garbage(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        with pytest.raises(
            ValueError, match="must be a number or a string .*optional suffix 'dB'"
        ):
            param.get_raw_value_for("loud")

    def test_get_raw_value_for_rejects_garbage_without_label(self):
        plugin = FakePlugin([FakeCppParameter("Gain", stepped_float(0, 1, 10))])
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        with pytest.raises(ValueError, match="must be a number or a string$"):
            param.get_raw_value_for("loud")

    def test_get_raw_value_for_rejects_out_of_range(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB"))]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=240)
        with pytest.raises(ValueError, match="out of range"):
            param.get_raw_value_for(13)
        with pytest.raises(ValueError, match="out of range"):
            param.get_raw_value_for(-13)

    def test_get_raw_value_for_falls_back_when_plugin_misparses_text(self):
        # The plugin claims every string maps to raw value 1.0, which is outside
        # the range we observed for the value. We should use our own range instead.
        plugin = FakePlugin(
            [
                FakeCppParameter(
                    "Gain", stepped_float(0, 10, 10), raw_for_text=lambda text: 1.0
                )
            ]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        raw = param.get_raw_value_for(5)
        assert 0.4 < raw < 0.6
        # But an in-range value reported by the plugin is trusted:
        assert param.get_raw_value_for(10) == 1.0


class TestBoolParameter:
    @pytest.mark.parametrize(
        "options", [["Off", "On"], ["no", "yes"], ["False", "True"]]
    )
    def test_infers_boolean(self, options):
        plugin = FakePlugin([FakeCppParameter("Bypass", choices(options))])
        param = AudioProcessorParameter(plugin, "Bypass", search_steps=100)
        assert param.type is bool
        assert param.min_value is False
        assert param.max_value is True
        assert param.step_size == 1
        assert param.valid_values == [False, True]
        assert 'boolean ("False" and "True")>' in repr(param)

    def test_get_raw_value_for(self):
        plugin = FakePlugin([FakeCppParameter("Bypass", choices(["Off", "On"]))])
        param = AudioProcessorParameter(plugin, "Bypass", search_steps=100)
        assert param.get_raw_value_for(True) == 1.0
        assert param.get_raw_value_for(False) == 0.0
        assert param.get_raw_value_for(WrappedBool(True)) == 1.0
        with pytest.raises(ValueError, match="should be a boolean"):
            param.get_raw_value_for("On")


class TestStringParameter:
    OPTIONS = ["Sine", "Square", "Saw"]

    def make(self, raw_for_text=None):
        plugin = FakePlugin(
            [
                FakeCppParameter(
                    "Waveform", choices(self.OPTIONS), raw_for_text=raw_for_text
                )
            ]
        )
        return AudioProcessorParameter(plugin, "Waveform", search_steps=300)

    def test_infers_string_type(self):
        param = self.make()
        assert param.type is str
        assert param.valid_values == self.OPTIONS
        assert param.min_value is None
        assert param.max_value is None
        assert param.step_size is None
        assert "(3 valid string values)>" in repr(param)

    def test_get_raw_value_for(self):
        param = self.make()
        assert param.get_raw_value_for("Sine") == 0.0
        assert 1 / 3 <= param.get_raw_value_for("Square") < 2 / 3
        assert param.get_raw_value_for("Saw") >= 2 / 3

    def test_get_raw_value_for_rejects_unknown_values(self):
        param = self.make()
        with pytest.raises(ValueError, match="not in list of valid values"):
            param.get_raw_value_for("Triangle")

    def test_get_raw_value_for_rejects_non_string_like(self):
        param = self.make()
        with pytest.raises(ValueError, match="should be a string"):
            param.get_raw_value_for(object())

    def test_get_raw_value_for_falls_back_when_plugin_misparses_text(self):
        param = self.make(raw_for_text=lambda text: 1.0)
        assert param.get_raw_value_for("Sine") == 0.0

    def test_two_string_values_that_are_not_booleans_stay_strings(self):
        plugin = FakePlugin([FakeCppParameter("Mode", choices(["Mono", "Stereo"]))])
        param = AudioProcessorParameter(plugin, "Mode", search_steps=100)
        assert param.type is str
        assert param.valid_values == ["Mono", "Stereo"]


class TestParameterEdgeCases:
    def test_slow_fetch_is_used_when_fast_path_returns_nothing_useful(self):
        plugin = FakePlugin(
            [FakeCppParameter("Gain", stepped_float(0, 10, 10, " dB"), slow_only=True)]
        )
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        assert param.type is float
        assert param.min_value == 0
        assert param.max_value == 10

    def test_raises_if_plugin_never_returns_a_string(self):
        plugin = FakePlugin([FakeCppParameter("Broken", lambda raw: None)])
        with pytest.raises(
            NotImplementedError, match="failed to return a valid string"
        ):
            AudioProcessorParameter(plugin, "Broken", search_steps=10)

    def test_raises_if_parameter_disappears(self):
        cpp = FakeCppParameter("Gain", stepped_float(0, 10, 10))
        plugin = FakePlugin([cpp])
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        del plugin._cpp_parameters["Gain"]
        with pytest.raises(RuntimeError, match="no longer available"):
            repr(param)
        # Attribute lookups swallow the RuntimeError and raise AttributeError instead:
        with pytest.raises(AttributeError):
            param.raw_value

    def test_attribute_access_is_forwarded_to_cpp_parameter(self):
        cpp = FakeCppParameter("Gain", stepped_float(0, 10, 10))
        plugin = FakePlugin([cpp])
        param = AudioProcessorParameter(plugin, "Gain", search_steps=100)
        assert param.name == "Gain"
        assert param.index == 0
        param.raw_value = 0.5
        assert cpp.raw_value == 0.5
        assert param.string_value == "5"
        # Attributes that don't exist on the C++ side are stored locally:
        param.custom_attribute = 123
        assert param.custom_attribute == 123
        assert not hasattr(cpp, "custom_attribute")
        with pytest.raises(AttributeError):
            param.definitely_not_an_attribute


class TestPythonExternalPluginMixin:
    def make_plugin(self):
        return FakePlugin(
            [
                FakeCppParameter("Gain", stepped_float(-12, 12, 24, " dB")),
                FakeCppParameter("Bypass", choices(["Off", "On"])),
                FakeCppParameter("Waveform", choices(["Sine", "Square", "Saw"])),
                FakeCppParameter("MIDI CC 1", stepped_float(0, 127, 127)),
                FakeCppParameter("P001", stepped_float(0, 127, 127)),
                FakeCppParameter("", stepped_float(0, 1, 1)),
            ]
        )

    def test_parameters_dict_is_read_only_and_filters_ignored_names(self):
        plugin = self.make_plugin()
        parameters = plugin.parameters
        assert set(parameters.keys()) == {"gain", "bypass", "waveform"}
        with pytest.raises(TypeError):
            parameters["gain"] = 1

    def test_parameters_are_cached(self):
        plugin = self.make_plugin()
        assert plugin.parameters["gain"] is plugin.parameters["gain"]

    def test_dir_includes_parameter_names(self):
        names = dir(self.make_plugin())
        assert "gain" in names
        assert "bypass" in names
        assert "waveform" in names

    def test_getattr_returns_typed_wrappers(self):
        plugin = self.make_plugin()
        assert isinstance(plugin.gain, float)
        assert plugin.gain == -12
        assert plugin.gain.label == "dB"
        assert bool(plugin.bypass) is False
        assert isinstance(plugin.waveform, str)
        assert plugin.waveform == "Sine"

    def test_setattr_sets_parameters(self):
        plugin = self.make_plugin()
        plugin.gain = 6
        assert plugin.gain == 6
        plugin.gain = "-6 dB"
        assert plugin.gain == -6
        plugin.bypass = True
        assert bool(plugin.bypass) is True
        plugin.waveform = "Saw"
        assert plugin.waveform == "Saw"

    def test_setattr_validates(self):
        plugin = self.make_plugin()
        with pytest.raises(ValueError, match="out of range"):
            plugin.gain = 100
        with pytest.raises(ValueError, match="not in list of valid values"):
            plugin.waveform = "Triangle"
        with pytest.raises(ValueError, match="should be a boolean"):
            plugin.bypass = "yes"

    def test_unknown_attributes_still_work_normally(self):
        plugin = self.make_plugin()
        plugin.not_a_parameter = 42
        assert plugin.not_a_parameter == 42
        with pytest.raises(AttributeError):
            plugin.also_not_a_parameter

    def test_set_initial_parameter_values(self):
        plugin = self.make_plugin()
        plugin.__set_initial_parameter_values__({"gain": 3, "waveform": "Square"})
        assert plugin.gain == 3
        assert plugin.waveform == "Square"

    def test_set_initial_parameter_values_accepts_none(self):
        plugin = self.make_plugin()
        plugin.__set_initial_parameter_values__(None)
        assert plugin.gain == -12

    def test_set_initial_parameter_values_rejects_non_dict(self):
        plugin = self.make_plugin()
        with pytest.raises(TypeError, match='pass "plugin_name=..."'):
            plugin.__set_initial_parameter_values__("My Plugin")

    def test_set_initial_parameter_values_rejects_unknown_parameter(self):
        plugin = self.make_plugin()
        plugin._parameter_weakrefs = plugin.parameters
        with pytest.raises(AttributeError, match='Parameter named "nope" not found'):
            plugin.__set_initial_parameter_values__({"nope": 1})

    def test_get_parameter_by_python_name_handles_removed_parameters(self):
        plugin = self.make_plugin()
        assert plugin._get_parameter_by_python_name("gain") is not None
        del plugin._cpp_parameters["Gain"]
        assert plugin._get_parameter_by_python_name("gain") is None

    def test_get_parameter_by_python_name_populates_cache_after_reset(self):
        plugin = self.make_plugin()
        plugin.parameters  # populate __python_to_cpp_names__
        plugin.__python_parameter_cache__ = {}
        param = plugin._get_parameter_by_python_name("gain")
        assert param is not None
        assert param.python_name == "gain"


class TestLoadPlugin:
    def test_missing_file_raises_import_error_listing_tried_formats(self, tmp_path):
        with pytest.raises(ImportError, match="Failed to load plugin as") as exc_info:
            load_plugin(str(tmp_path / "does_not_exist.vst3"))
        for plugin_class in pedalboard._pedalboard._AVAILABLE_PLUGIN_CLASSES:
            assert plugin_class.__name__ in str(exc_info.value)

    def test_no_available_plugin_classes(self, monkeypatch):
        monkeypatch.setattr(pedalboard._pedalboard, "_AVAILABLE_PLUGIN_CLASSES", [])
        with pytest.raises(ImportError, match="no supported external plugin types"):
            load_plugin("anything.vst3")

    def test_error_message_joins_three_or_more_formats(self, monkeypatch):
        class FakeA:
            def __init__(self, **kwargs):
                raise ImportError("a failed")

        class FakeB(FakeA):
            pass

        class FakeC(FakeA):
            pass

        monkeypatch.setattr(
            pedalboard._pedalboard, "_AVAILABLE_PLUGIN_CLASSES", [FakeA, FakeB, FakeC]
        )
        with pytest.raises(ImportError) as exc_info:
            load_plugin("anything.vst3")
        message = str(exc_info.value)
        assert "FakeA, FakeB, or FakeC" in message
        assert "FakeA: a failed" in message

    def test_non_import_errors_propagate(self, monkeypatch):
        class Explodes:
            def __init__(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            pedalboard._pedalboard, "_AVAILABLE_PLUGIN_CLASSES", [Explodes]
        )
        with pytest.raises(RuntimeError, match="boom"):
            load_plugin("anything.vst3")

    def test_arguments_are_forwarded(self, monkeypatch):
        received: Dict[str, object] = {}

        class Records:
            def __init__(self, **kwargs):
                received.update(kwargs)

        monkeypatch.setattr(
            pedalboard._pedalboard, "_AVAILABLE_PLUGIN_CLASSES", [Records]
        )
        result = load_plugin(
            "a.vst3", {"gain": 1}, plugin_name="X", initialization_timeout=2.5
        )
        assert isinstance(result, Records)
        assert received == {
            "path_to_plugin_file": "a.vst3",
            "parameter_values": {"gain": 1},
            "plugin_name": "X",
            "initialization_timeout": 2.5,
        }
