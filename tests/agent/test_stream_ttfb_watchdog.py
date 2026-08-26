"""Regression tests for the no-first-byte TTFB watchdog (generic streaming path).

A provider can accept a connection without emitting a stream event.
The stale-stream detector is deliberately scaled for reasoning models, so
this watchdog supplies a separate cutoff for retrying a dead connection.

These tests pin the production TTFB resolution and kill-predicate helpers
without reproducing their implementation in the test suite.

1. Cloud providers get a 120s default no-first-byte cutoff.
2. Local endpoints disable the watchdog (prefill can take minutes).
3. ``HERMES_STREAM_TTFB_TIMEOUT_SECONDS=0`` disables it entirely.
4. Large contexts scale the cutoff up (240s / 300s) so healthy
   backend admission / prompt prefill is not aborted.
5. The cutoff never exceeds the stale-stream patience: the stale
   detector owns the terminal kill + diagnostics for a connection
   that delivered bytes then wedged.
6. The kill predicate fires only when NO stream event has been seen
   and the elapsed time is past the cutoff.
"""

from __future__ import annotations

from agent.chat_completion_helpers import (
    resolve_stream_ttfb_timeout,
    ttfb_kill_should_fire,
)


CLOUD_URLS = [
    "https://api.githubcopilot.com",
    "https://api.openai.com",
    "https://openrouter.ai/api",
    "https://api.anthropic.com",
    "https://api.example.com/v1",
]


class TestTtfbResolution:
    def test_cloud_default_is_120s(self):
        for url in CLOUD_URLS:
            assert resolve_stream_ttfb_timeout(url, est_tokens=0, stale_timeout=600.0) == 120.0

    def test_local_endpoint_disables_watchdog(self):
        assert resolve_stream_ttfb_timeout("http://localhost:11434", est_tokens=0, stale_timeout=float("inf")) == float("inf")
        assert resolve_stream_ttfb_timeout("http://127.0.0.1:8080", est_tokens=0, stale_timeout=float("inf")) == float("inf")

    def test_env_zero_disables_watchdog(self):
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=600.0, env_value="0") == float("inf")

    def test_env_override_is_respected(self):
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=600.0, env_value="45") == 45.0

    def test_empty_env_falls_back_to_default(self):
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=600.0, env_value="") == 120.0

    def test_garbage_env_falls_back_to_default(self):
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=600.0, env_value="not-a-number") == 120.0

    def test_large_context_scales_up(self):
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=60_000, stale_timeout=600.0) == 240.0
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=150_000, stale_timeout=600.0) == 300.0

    def test_never_exceeds_stale_patience(self):
        # A 120s default TTFB with a 90s stale timeout must clamp to 90s.
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=90.0) == 90.0
        # A scaled-up TTFB (300s) with a 180s stale timeout must clamp to 180s.
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=150_000, stale_timeout=180.0) == 180.0

    def test_infinite_stale_keeps_ttfb_finite(self):
        # Local endpoints disable BOTH, but a cloud endpoint with an
        # infinite stale value (explicit user config) still gets the
        # finite TTFB cutoff — the watchdog is the only bound left.
        assert resolve_stream_ttfb_timeout("https://api.openai.com", est_tokens=0, stale_timeout=float("inf")) == 120.0


class TestTtfbKillPredicate:
    def test_no_first_byte_past_cutoff_fires(self):
        assert ttfb_kill_should_fire(False, elapsed=121.0, ttfb_timeout=120.0) is True

    def test_no_first_byte_before_cutoff_waits(self):
        assert ttfb_kill_should_fire(False, elapsed=119.0, ttfb_timeout=120.0) is False

    def test_first_byte_seen_disarms(self):
        assert ttfb_kill_should_fire(True, elapsed=500.0, ttfb_timeout=120.0) is False

    def test_disabled_watchdog_never_fires(self):
        assert ttfb_kill_should_fire(False, elapsed=9999.0, ttfb_timeout=float("inf")) is False

    def test_kill_reset_suppresses_repeat_until_next_attempt(self):
        assert ttfb_kill_should_fire(False, elapsed=121.0, ttfb_timeout=120.0) is True
        # The production kill branch marks the attempt as seen after resetting
        # its timer; a fresh attempt clears the flag before polling resumes.
        assert ttfb_kill_should_fire(True, elapsed=1.0, ttfb_timeout=120.0) is False
        assert ttfb_kill_should_fire(False, elapsed=121.0, ttfb_timeout=120.0) is True
