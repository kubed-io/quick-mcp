"""The telemetry settings: every signal off until a config turns it on."""

from __future__ import annotations

from pydantic import BaseModel

from kubed.mcp import TelemetrySettings


def test_settings_default_to_every_signal_off() -> None:
    settings = TelemetrySettings()
    assert settings.profiles is None
    assert settings.profile_cpu is True
    assert settings.profile_memory is True
    assert settings.otel is False
    assert settings.tags == {}


def test_settings_embed_as_a_configs_telemetry_section() -> None:
    class Config(BaseModel):
        telemetry: TelemetrySettings = TelemetrySettings()

    config = Config.model_validate(
        {
            "telemetry": {
                "profiles": "http://pyroscope.example.com:4040",
                "tags": {"team": "a"},
            }
        }
    )
    assert config.telemetry.profiles == "http://pyroscope.example.com:4040"
    assert config.telemetry.tags == {"team": "a"}
    assert Config().telemetry.tags is not Config().telemetry.tags
