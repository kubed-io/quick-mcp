"""Where a config's values come from: its items, their names, and its secrets."""

from __future__ import annotations

import json
import re
from typing import Annotated, Optional

import pytest
from pydantic import Secret, SecretStr, create_model, field_validator

from kubed.mcp._config import (
    Config,
    Field,
    NoDecode,
    _compose,
    _load,
    config,
    config_schema,
)

pytestmark = pytest.mark.unit


# -- items


@config()
class Serve(Config):
    port: int = Field(8000, env="PORT", arg="--port")
    name: str = "flow"


@config("recording")
class Rec(Config):
    inbox: str | None = Field(None, env="RECORDING_INBOX", arg="--recording-inbox")
    tags: dict[str, str] = Field(
        default_factory=dict, env="RECORDING_TAGS", arg="--recording-tags"
    )
    hosts: list[str] = Field(default_factory=list, env="RECORDING_HOSTS")
    raw: Annotated[list[str], NoDecode] = Field(
        default_factory=list, env="RECORDING_RAW"
    )
    token: SecretStr | None = Field(
        None, env="RECORDING_TOKEN", arg="--recording-token"
    )
    keys: Secret[dict[str, int]] | None = Field(None, env="RECORDING_KEYS")

    @field_validator("raw", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        return value.split(",") if isinstance(value, str) else value


class Limits(Config):
    burst: int = 1
    rate: int = Field(2, env="TELEMETRY_RATE")


@config("telemetry")
class Tel(Config):
    limits: Limits = Field(Limits(burst=5), env="TELEMETRY_LIMITS")


class Linked(Config):
    val: int = 0
    next: Linked | None = None


@config("chain")
class Chain(Config):
    head: Linked = Linked()


class Db(Config):
    host: str = Field("", env="DB_HOST")


@config("dbs")
class Dbs(Config):
    primary: Db = Db()
    replica: Db = Db()


class Server(Config):
    host: str = Field("", env="SERVER_HOST")


@config("fleet")
class Fleet(Config):
    servers: list[Server] = Field(default_factory=list)


class Pod(Config):
    server: Server = Server()


@config("pods")
class Pods(Config):
    pods: list[Pod] = Field(default_factory=list)


@config("both")
class Both(Config):
    one: Server = Server()
    many: list[Server] = Field(default_factory=list)


@config("marked")
class Marked(Config):
    t: SecretStr = Field(SecretStr("dflt"), sensitive=True)


@config("listed")
class Listed(Config):
    ts: list[SecretStr] = Field(default_factory=list)
    by_name: dict[str, SecretStr] = Field(default_factory=dict)


@config("plain")
class PlainMarked(Config):
    token: str = Field("x", sensitive=True)


@config("nested")
class NestedMark(Config):
    t: Optional[Annotated[SecretStr, Field(sensitive=True)]] = None  # noqa: UP045


@config("nestedenv")
class NestedEnv(Config):
    xs: list[Annotated[str, Field(env="X")]] = Field(default_factory=list)


@config("mixed")
class Mixed(Config):
    either: int | list[int] = 0
    maybe: tuple[int, ...] | None = None


@config("aliased")
class Aliased(Config):
    inbox: str = Field("", validation_alias="Inbox")


@config("dupa")
class DupEnvA(Config):
    x: int = Field(0, env="DUP")


@config("dupb")
class DupEnvB(Config):
    y: int = Field(0, env="DUP")


@config("flaga")
class DupArgA(Config):
    x: int = Field(0, arg="--dup")


@config("flagb")
class DupArgB(Config):
    y: int = Field(0, arg="--dup")


def paths(*sections):
    return {i.path: i for i in _compose([sections]).items}


def test_items_follow_model_fields_and_stop_at_everything_else() -> None:
    items = paths(Serve, Rec, Tel)
    assert list(items) == [
        "port",
        "name",
        "recording",
        "recording.inbox",
        "recording.tags",
        "recording.hosts",
        "recording.raw",
        "recording.token",
        "recording.keys",
        "telemetry",
        "telemetry.limits",
        "telemetry.limits.burst",
        "telemetry.limits.rate",
    ]
    assert [p for p, i in items.items() if i.leaf] == [
        "port",
        "name",
        "recording.inbox",
        "recording.tags",
        "recording.hosts",
        "recording.raw",
        "recording.token",
        "recording.keys",
        "telemetry.limits.burst",
        "telemetry.limits.rate",
    ]
    limits = items["telemetry.limits"]
    assert not limits.leaf and limits.env == "TELEMETRY_LIMITS"
    assert items["port"].env == "PORT" and items["port"].arg == "--port"
    assert items["name"].env is None and items["name"].arg is None
    assert items["telemetry.limits.rate"].loc == ("telemetry", "limits", "rate")


def test_an_items_loc_is_its_alias_and_its_attrs_are_its_names() -> None:
    item = paths(Aliased)["aliased.Inbox"]
    assert item.loc == ("aliased", "Inbox") and item.attrs == ("aliased", "inbox")


@pytest.mark.parametrize(
    "path, takes",
    [
        ("port", "string"),
        ("recording.inbox", "string"),
        ("recording.tags", "object"),
        ("recording.hosts", "array"),
        ("recording.raw", "string"),
        ("recording.token", "string"),
        ("recording.keys", "object"),
        ("telemetry.limits", "object"),
    ],
)
def test_each_item_knows_how_its_raw_value_decodes(path, takes) -> None:
    assert paths(Serve, Rec, Tel)[path].takes == takes


def test_a_union_takes_a_kind_only_when_every_member_has_it() -> None:
    items = paths(Mixed)
    assert items["mixed.either"].takes == "string"  # int | list[int]
    assert items["mixed.maybe"].takes == "array"  # tuple[int, ...] | None


def test_a_secret_typed_field_is_sensitive_marked_or_not() -> None:
    items = paths(Rec)
    assert items["recording.token"].sensitive and items["recording.keys"].sensitive
    assert not items["recording.inbox"].sensitive
    assert paths(Marked)["marked.t"].sensitive
    listed = paths(Listed)
    assert listed["listed.ts"].sensitive and listed["listed.by_name"].sensitive


def test_sensitive_on_a_type_that_is_not_a_secret_is_refused() -> None:
    with pytest.raises(
        TypeError,
        match=r"plain\.token: sensitive=True needs a secret type: use SecretStr",
    ):
        _compose([[PlainMarked]])


@pytest.mark.parametrize(
    "section, path", [(NestedMark, "nested.t"), (NestedEnv, "nestedenv.xs")]
)
def test_an_item_keyword_inside_an_annotation_is_refused(section, path) -> None:
    with pytest.raises(
        TypeError, match=rf"{re.escape(path)}: .*Field\(\.\.\.\) on the field itself"
    ):
        _compose([[section]])


@pytest.mark.parametrize(
    "field, match",
    [({"arg": "port"}, "--"), ({"arg": "--"}, "--"), ({"env": "A=B"}, "=")],
)
def test_a_bad_flag_or_env_name_is_refused(field, match) -> None:
    section = config("bad")(
        create_model("Bad", __base__=Config, x=(int, Field(0, **field)))
    )
    with pytest.raises(TypeError, match=rf"bad\.x: .*{re.escape(match)}"):
        _compose([[section]])


def test_a_name_declared_twice_names_both_paths() -> None:
    with pytest.raises(
        ValueError, match=r"DB_HOST.*dbs\.primary\.host.*dbs\.replica\.host"
    ):
        _compose([[Dbs]])
    with pytest.raises(ValueError, match=r"DUP.*dupa\.x.*dupb\.y"):
        _compose([[DupEnvA, DupEnvB]])
    with pytest.raises(ValueError, match=r"--dup.*flaga\.x.*flagb\.y"):
        _compose([[DupArgA, DupArgB]])


@pytest.mark.parametrize("field", [{"env": "CONFIG_FILE"}, {"arg": "--config-file"}])
def test_quick_mcps_own_names_are_reserved(field) -> None:
    section = config("own")(
        create_model("Own", __base__=Config, x=(str, Field("", **field)))
    )
    with pytest.raises(ValueError, match=r"own\.x: .*config file"):
        _compose([[section]])


def test_a_name_inside_a_list_of_models_is_refused() -> None:
    with pytest.raises(
        TypeError, match=r"fleet\.servers.*SERVER_HOST|SERVER_HOST.*fleet\.servers"
    ):
        _compose([[Fleet]])
    # a model held below a list, at any depth
    with pytest.raises(
        TypeError, match=r"pods\.pods.*SERVER_HOST|SERVER_HOST.*pods\.pods"
    ):
        _compose([[Pods]])


def test_a_name_on_a_model_also_reached_directly_is_kept() -> None:
    items = paths(Both)
    assert items["both.one.host"].env == "SERVER_HOST" and items["both.many"].leaf


def test_a_self_referencing_model_ends_the_walk() -> None:
    items = paths(Chain)
    assert "chain.head.val" in items and items["chain.head.next"].leaf


def test_the_schema_marks_every_secret_and_writes_no_default() -> None:
    schema = config_schema(Rec, Marked)
    props = schema["$defs"]["Rec"]["properties"]
    for name in ("token", "keys"):
        assert props[name]["sensitive"] is True and "default" not in props[name]
    assert "sensitive" not in props["inbox"]
    marked = schema["$defs"]["Marked"]["properties"]["t"]
    assert marked["sensitive"] is True and "default" not in marked
    assert "dflt" not in json.dumps(config_schema(Marked))  # Marked's default is dflt


def test_a_secret_is_masked_in_every_repr_and_dump() -> None:
    loaded = _load([[Rec]], {"recording": {"token": "s3cret", "keys": {"k": 1}}}).config
    for text in (
        repr(loaded),
        str(loaded),
        repr(loaded.recording),
        loaded.model_dump_json(),
        loaded.recording.model_dump_json(),
    ):
        assert "s3cret" not in text and '"k"' not in text
    assert loaded.recording.token.get_secret_value() == "s3cret"
