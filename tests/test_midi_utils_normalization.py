#! /usr/bin/env python
#
# Copyright 2023 Spotify AB
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

"""
Unit tests for the duck-typed MIDI message normalization in ``pedalboard/midi_utils.py``.
"""

import pytest

from pedalboard.midi_utils import (
    normalize_midi_messages,
    parse_midi_message_part,
    parse_midi_message_string,
)

NOTE_ON = bytes([0x90, 60, 100])
NOTE_OFF = bytes([0x80, 60, 0])


class TestParseMidiMessagePart:
    def test_int_passthrough(self):
        assert parse_midi_message_part(144) == 144

    def test_single_byte(self):
        assert parse_midi_message_part(b"\x90") == 0x90

    def test_multi_byte_bytes_is_rejected(self):
        with pytest.raises(NotImplementedError, match="Not sure how to interpret"):
            parse_midi_message_part(b"\x90\x3c")

    def test_numeric_string(self):
        assert parse_midi_message_part("144") == 144

    def test_non_numeric_string_is_rejected(self):
        with pytest.raises(
            NotImplementedError, match="Not sure how to interpret"
        ) as exc_info:
            parse_midi_message_part("note_on")
        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.parametrize("value", [1.5, None, [1]])
    def test_unsupported_types_are_rejected(self, value):
        with pytest.raises(NotImplementedError, match="must currently be bytes"):
            parse_midi_message_part(value)


def test_parse_midi_message_string_is_not_implemented():
    with pytest.raises(NotImplementedError, match="must currently be bytes"):
        parse_midi_message_string("note_on 60 100")


class TestNormalizeMidiMessages:
    def test_empty_input(self):
        assert normalize_midi_messages([]) == []

    def test_tuples_of_bytes(self):
        assert normalize_midi_messages([(NOTE_ON, 0.0), (NOTE_OFF, 1.0)]) == [
            (NOTE_ON, 0.0),
            (NOTE_OFF, 1.0),
        ]

    def test_lists_are_accepted_as_message_containers(self):
        assert normalize_midi_messages([[NOTE_ON, 0.5]]) == [(NOTE_ON, 0.5)]

    def test_lists_of_ints_are_converted_to_bytes(self):
        assert normalize_midi_messages([([0x90, 60, 100], 0.0)]) == [(NOTE_ON, 0.0)]

    def test_lists_of_mixed_parts_are_converted_to_bytes(self):
        assert normalize_midi_messages([([b"\x90", "60", 100], 0.0)]) == [
            (NOTE_ON, 0.0)
        ]

    def test_bytearray_and_other_bytes_like_values_are_converted(self):
        assert normalize_midi_messages([(bytearray(NOTE_ON), 0.0)]) == [(NOTE_ON, 0.0)]

    def test_string_messages_are_rejected(self):
        with pytest.raises(NotImplementedError):
            normalize_midi_messages([("note_on 60 100", 0.0)])

    def test_objects_with_bytes_and_time_attributes(self):
        class Message:
            time = 2.0

            def bytes(self):
                return [0x90, 60, 100]

        assert normalize_midi_messages([Message()]) == [(NOTE_ON, 2.0)]

    def test_generators_are_accepted(self):
        result = normalize_midi_messages((NOTE_ON, float(i)) for i in range(3))
        assert result == [(NOTE_ON, 0.0), (NOTE_ON, 1.0), (NOTE_ON, 2.0)]

    def test_unrecognized_entries_are_silently_dropped(self):
        assert normalize_midi_messages([42, "nope", (1, 2, 3), (NOTE_ON, 0.0)]) == [
            (NOTE_ON, 0.0)
        ]

    def test_absolute_timestamps_are_accepted_for_long_streams(self):
        messages = [(NOTE_ON, i * 0.01) for i in range(500)]
        assert len(normalize_midi_messages(messages)) == 500

    def test_identical_timestamps_are_accepted_for_long_streams(self):
        # All-identical timestamps are not treated as deltas (only one distinct value).
        messages = [(NOTE_ON, 0.0) for _ in range(500)]
        assert len(normalize_midi_messages(messages)) == 500

    def test_short_streams_with_repeated_timestamps_are_accepted(self):
        messages = [(NOTE_ON, 0.0) for _ in range(50)] + [
            (NOTE_OFF, 1.0) for _ in range(50)
        ]
        assert len(normalize_midi_messages(messages)) == 100

    def test_probable_delta_timestamps_are_rejected(self):
        # 101+ events sharing a timestamp, with more than one distinct timestamp,
        # looks like delta-encoded time.
        messages = [(NOTE_ON, 0.0)] + [(NOTE_ON, 0.25) for _ in range(150)]
        with pytest.raises(ValueError, match="absolute values") as exc_info:
            normalize_midi_messages(messages)
        assert "150 events at timestamp 0.25" in str(exc_info.value)

    def test_repeated_timestamps_below_threshold_are_accepted(self):
        messages = [(NOTE_ON, 0.0)] + [(NOTE_ON, 0.25) for _ in range(100)]
        assert len(normalize_midi_messages(messages)) == 101

    def test_delta_detection_requires_more_than_100_messages(self):
        # Exactly 100 messages never trigger delta detection, even if 99 of them
        # share a timestamp; only the messages that survive parsing are counted.
        messages = [(NOTE_ON, 0.0)] + [(NOTE_ON, 0.25) for _ in range(99)]
        assert len(normalize_midi_messages(messages)) == 100
        assert len(normalize_midi_messages(messages + ["dropped"] * 500)) == 100
