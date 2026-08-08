"""Tests for ``rank.decay`` — the ``rank_by`` builder and its unit conversions.

Three things here are silent when wrong, which is why the builder exists at all rather than
callers writing the tagged dict:

* **The wire name is ``lambda``**, a reserved word in Python, so the argument is ``lambda_``
  and the rename has to happen exactly once, here.
* **Every unset knob must be omitted**, not defaulted in Python: the server owns the
  defaults (a week's half-life, ``missing = 1.0`` so an undated record is *not* penalized),
  and restating them would fork the contract the day one changes.
* **The unit is epoch milliseconds.** A ``datetime``/``timedelta`` is converted by exact
  integer arithmetic, never through ``timestamp()``, which routes a large instant through a
  float; and a naive ``datetime`` is refused rather than assumed to be UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nidus import rank

# 2023-11-14T22:13:20Z, the round number the rest of the suite uses for instants.
ORIGIN_MS = 1700000000000
ORIGIN = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)


def test_decay_is_tagged_and_carries_only_what_was_asked_for() -> None:
    """The minimal expression: a field and an origin, every other knob the server's."""
    assert rank.decay("updated_at", ORIGIN_MS) == {
        "Decay": {"field": "updated_at", "origin": ORIGIN_MS}
    }


def test_decay_sends_every_knob_under_the_servers_own_name() -> None:
    """``lambda_`` travels as ``lambda`` — the one place that rename may happen."""
    assert rank.decay("ts", ORIGIN_MS, scale=86_400_000, decay=0.9, lambda_=0.25, missing=0.0) == {
        "Decay": {
            "field": "ts",
            "origin": ORIGIN_MS,
            "scale": 86_400_000,
            "decay": 0.9,
            "lambda": 0.25,
            "missing": 0.0,
        }
    }


def test_an_explicit_zero_knob_is_sent_rather_than_pruned() -> None:
    """``missing=0.0`` (penalize an undated record fully) is a real value, not "unset"."""
    assert rank.decay("ts", ORIGIN_MS, missing=0.0)["Decay"]["missing"] == 0.0
    assert rank.decay("ts", ORIGIN_MS, lambda_=0.0)["Decay"]["lambda"] == 0.0
    assert "missing" not in rank.decay("ts", ORIGIN_MS)["Decay"]


def test_an_origin_may_be_a_datetime_or_raw_epoch_milliseconds() -> None:
    """Both spellings reach the same number, so neither is the "real" one."""
    assert rank.decay("ts", ORIGIN) == rank.decay("ts", ORIGIN_MS)
    assert rank.decay("ts", ORIGIN)["Decay"]["origin"] == ORIGIN_MS


def test_a_scale_may_be_a_timedelta() -> None:
    """The wire unit is milliseconds; a ``timedelta`` is the readable way to say a week."""
    assert rank.decay("ts", ORIGIN_MS, scale=timedelta(days=7)) == rank.decay(
        "ts", ORIGIN_MS, scale=7 * 86_400_000
    )
    assert rank.decay("ts", ORIGIN_MS, scale=timedelta(milliseconds=1))["Decay"]["scale"] == 1
    # Sub-millisecond precision is truncated, as it is for a `DateTime` attribute.
    assert rank.decay("ts", ORIGIN_MS, scale=timedelta(microseconds=1500))["Decay"]["scale"] == 1


def test_a_naive_origin_is_refused_rather_than_assumed_to_be_utc() -> None:
    """The same rule ``v.datetime`` applies: guessing a timezone is wrong by hours."""
    with pytest.raises(ValueError, match="aware datetime"):
        rank.decay("ts", datetime(2023, 11, 14))


def test_a_non_integer_scale_names_the_argument() -> None:
    """A ``float`` here is almost always seconds; the server's ``scale`` is an ``i64`` of ms."""
    with pytest.raises(TypeError, match=r"rank\.decay\(scale"):
        rank.decay("ts", ORIGIN_MS, scale=1.5)  # type: ignore[arg-type]
