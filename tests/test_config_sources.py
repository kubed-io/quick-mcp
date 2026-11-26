"""Where a config's values come from: its items, their names, and its secrets."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from types import SimpleNamespace
from typing import Annotated, Any, Optional

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Json,
    Secret,
    SecretStr,
    create_model,
    field_serializer,
    field_validator,
    model_validator,
)

from kubed.mcp import QuickMCP
from kubed.mcp._config import (
    Config,
    Field,
    NoDecode,
    Source,
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


@pytest.mark.parametrize(
    "alias",
    [AliasChoices("x", "xx"), AliasPath("x", 0), AliasChoices(AliasPath("x", 0))],
)
@pytest.mark.parametrize("name", [{"env": "CHOSEN_X"}, {"arg": "--chosen-x"}])
def test_a_named_item_with_an_alias_that_is_not_a_string_is_refused(
    alias, name
) -> None:
    section = config("chosen")(
        create_model(
            "Chosen", __base__=Config, x=(int, Field(0, validation_alias=alias, **name))
        )
    )
    with pytest.raises(
        TypeError,
        match=r"^chosen\.x: a named item takes a plain string alias or none",
    ):
        _compose([[section]])


@config("routes")
class Routes(Config):
    a: Secret[dict[str, int]] | None = Field(
        None, validation_alias=AliasChoices("k", "keys")
    )
    b: Secret[dict[str, int]] | None = Field(
        None, validation_alias=AliasChoices(AliasPath("bb", 0), "z")
    )
    c: Secret[dict[str, int]] | None = Field(None, validation_alias=AliasPath("cc", 0))
    d: Secret[dict[str, int]] | None = Field(None, validation_alias=AliasPath("dd"))


def test_the_schema_marks_a_secret_under_the_key_pydantic_writes() -> None:
    props = config_schema(Routes)["$defs"]["Routes"]["properties"]
    assert list(props) == ["k", "z", "c", "d"]
    for prop in props.values():
        assert prop["sensitive"] is True and "default" not in prop


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


def load(*sections, argv=None, environ=None):
    return _load(
        [sections or (Serve, Rec, Tel)],
        None,
        argv=argv,
        environ={} if environ is None else environ,
    )


# -- sources


def test_each_source_beats_the_one_below(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("port: 8001\n")
    env = {"CONFIG_FILE": str(f)}
    assert load().config.port == 8000
    assert load(environ=env).config.port == 8001
    assert load(environ={**env, "PORT": "8002"}).config.port == 8002
    assert (
        load(environ={**env, "PORT": "8002"}, argv=["--port", "8003"]).config.port
        == 8003
    )


def test_a_field_without_env_ignores_a_same_named_var_and_names_are_case_sensitive() -> (
    None
):
    loaded = load(environ={"NAME": "x", "name": "x", "port": "1"})
    assert loaded.config.name == "flow" and loaded.config.port == 8000


def test_env_on_a_nested_key_keeps_the_files_siblings(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("telemetry:\n  limits:\n    burst: 7\n")
    limits = load(
        environ={"CONFIG_FILE": str(f), "TELEMETRY_RATE": "9"}
    ).config.telemetry.limits
    assert (limits.burst, limits.rate) == (7, 9)


def test_a_default_instance_keeps_its_values_under_a_layer() -> None:
    limits = load(environ={"TELEMETRY_RATE": "9"}).config.telemetry.limits
    assert (limits.burst, limits.rate) == (5, 9)
    assert (
        _load(
            [[Tel]], {"telemetry": {"limits": {"rate": 9}}}
        ).config.telemetry.limits.burst
        == 5
    )


@config("made")
class Made(Config):
    limits: Limits = Field(default_factory=lambda: Limits(burst=5))


def _never(data: dict[str, Any]) -> Limits:
    raise AssertionError("a factory that takes the validated data was called")


@config("fromdata")
class FromData(Config):
    n: int = 0
    limits: Limits = Field(default_factory=_never)


def test_a_factory_default_keeps_its_values_under_a_layer() -> None:
    limits = load(Made, environ={"TELEMETRY_RATE": "9"}).config.made.limits
    assert (limits.burst, limits.rate) == (5, 9)
    assert load(Made).config.made.limits.burst == 5


def test_a_factory_that_takes_the_validated_data_is_never_called() -> None:
    limits = load(FromData, environ={"TELEMETRY_RATE": "9"}).config.fromdata.limits
    assert (limits.burst, limits.rate) == (1, 9)


def test_a_nested_name_beats_the_model_field_that_holds_it() -> None:
    env = {"TELEMETRY_LIMITS": '{"burst": 3, "rate": 4}', "TELEMETRY_RATE": "9"}
    limits = load(environ=env).config.telemetry.limits
    assert (limits.burst, limits.rate) == (3, 9)


def test_a_list_replaces_whole(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("recording:\n  hosts: [a, b]\n")
    environ = {"CONFIG_FILE": str(f), "RECORDING_HOSTS": '["c"]'}
    assert load(environ=environ).config.recording.hosts == ["c"]


def test_values_in_code_read_no_env_and_no_file(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "1")
    monkeypatch.setenv("CONFIG_FILE", "/does/not/exist.yaml")
    assert _load([[Serve]], {"port": 2}).config.port == 2


@pytest.mark.parametrize("extra", [{"argv": []}, {"environ": {}}])
def test_config_with_argv_or_environ_is_a_type_error(extra) -> None:
    with pytest.raises(TypeError, match=r"config= reads no flags or env vars"):
        _load([[Serve]], {"port": 2}, **extra)


def test_environ_none_reads_the_process_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "8004")
    assert _load([[Serve]], None).config.port == 8004


@pytest.mark.parametrize("environ", [[("PORT", "1")], "PORT=1"])
def test_environ_must_be_a_mapping(environ) -> None:
    with pytest.raises(TypeError, match=r"environ= takes a mapping .* os\.environ"):
        _load([[Serve]], None, environ=environ)


# -- blanks


def test_blank_values_never_override_a_lower_source(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("port: 8001\nrecording:\n  inbox: /in\n  tags: {a: b}\n")
    loaded = load(
        environ={"CONFIG_FILE": str(f), "PORT": "  ", "RECORDING_TAGS": ""},
        argv=["--port", "", "--recording-inbox", " "],
    )
    assert loaded.config.port == 8001 and loaded.config.recording.inbox == "/in"
    assert loaded.config.recording.tags == {"a": "b"}


def test_blank_and_null_file_entries_are_unset_at_any_depth(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("port:\nname: ' '\ntelemetry:\n  limits:\n    burst: ~\n")
    loaded = load(environ={"CONFIG_FILE": str(f)})
    assert (
        loaded.config.port,
        loaded.config.name,
        loaded.config.telemetry.limits.burst,
    ) == (8000, "flow", 5)


def test_a_blank_json_env_var_is_unset_not_a_decode_error() -> None:
    environ = {"RECORDING_TAGS": "  ", "RECORDING_HOSTS": ""}
    assert load(environ=environ).config.recording.tags == {}


def test_a_blank_inside_a_list_is_left_to_validation(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("recording:\n  hosts: ['', a]\n")
    assert load(environ={"CONFIG_FILE": str(f)}).config.recording.hosts == ["", "a"]


# -- the file


def _file_error(path) -> str:
    """The load's message for a named file, with an env error beside it that must not show."""
    with pytest.raises(ValueError) as caught:
        load(environ={"CONFIG_FILE": str(path), "PORT": "not a number"})
    return str(caught.value)


def test_a_missing_file_is_the_only_error(tmp_path) -> None:
    p = tmp_path / "nope.yaml"
    assert (
        _file_error(p) == f"the config did not load:\n  config file {p} does not exist"
    )


@pytest.mark.parametrize(
    "content, tail",
    [
        (None, r"is not a file"),  # None: CONFIG_FILE names tmp_path itself
        (b"\xff\xfe", r"cannot be read: UnicodeDecodeError"),
        (b"port: [1\nname: s3cret\n", r"is not valid YAML at line \d+, column \d+"),
        (b"- 1\n", r"must be a mapping, not a list"),
        (b"when: 2020-02-30\n", r"is not valid YAML"),  # a constructor's ValueError
        (b"port: !!timestamp 's3cret'\n", r"is not valid YAML"),  # AttributeError
        (b"a: &x {b: *x, s3cret: 1}\n", r"is not valid YAML"),  # refers to itself
    ],
)
def test_a_named_file_that_cannot_be_used_stops_the_load(
    tmp_path, content, tail
) -> None:
    path = tmp_path
    if content is not None:
        path = tmp_path / "c.yaml"
        path.write_bytes(content)
    message = _file_error(path)
    assert re.fullmatch(
        rf"the config did not load:\n  config file {re.escape(str(path))} {tail}",
        message,
    )
    assert "s3cret" not in message


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_file_names_the_error_class(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("port: 1\n")
    p.chmod(0)
    assert _file_error(p).endswith(f"config file {p} cannot be read: PermissionError")


def test_an_empty_file_is_no_values(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("")
    assert load(environ={"CONFIG_FILE": str(f)}).config.port == 8000


def test_the_flag_names_the_file_over_the_env_var_and_a_blank_flag_is_unset(
    tmp_path,
) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("port: 1\n")
    b.write_text("port: 2\n")
    env = {"CONFIG_FILE": str(a)}
    assert load(environ=env, argv=["--config-file", str(b)]).config.port == 2
    assert load(environ=env, argv=["--config-file", " "]).config.port == 1
    assert load(environ={"CONFIG_FILE": ""}).config.port == 8000


# -- argv


@config("dashes")
class Dashes(Config):
    a: int = Field(0, arg="--a-b")
    b: int = Field(0, arg="--a_b")


def test_argv_none_reads_no_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["-c", "x", "--config-file", "y", "--port", "1"])
    assert load().config.port == 8000


@pytest.mark.parametrize(
    "argv, flag",
    [
        (["--tokn", "s3cret"], "--tokn"),
        (["--tokn=s3cret"], "--tokn"),
        (["--po", "1"], "--po"),
        (["--tokn s3cret"], "--tokn"),
        (["--tokn", "-s3cret"], "--tokn"),
    ],
)
def test_an_unknown_or_abbreviated_flag_is_a_collected_error(argv, flag) -> None:
    with pytest.raises(ValueError) as caught:
        load(argv=argv)
    message = str(caught.value)
    assert f"  the arguments (arg {flag}): no item takes this flag" in message
    assert "s3cret" not in message and "positional" not in message


@pytest.mark.parametrize(
    "argv, count",
    [(["-v"], 1), (["---s3cret"], 1), (["-"], 1), (["--tokn=x", "-s3cret"], 1)],
)
def test_a_short_or_malformed_flag_is_counted_as_a_positional(argv, count) -> None:
    with pytest.raises(ValueError) as caught:
        load(argv=argv)
    message = str(caught.value)
    assert (
        f"  the arguments (arg): no item takes a positional argument, got {count}; "
        "a server passes argv=sys.argv[1:]" in message
    )
    assert "s3cret" not in message and "-v" not in message
    assert ("(arg --tokn): no item takes this flag" in message) == ("--tokn=x" in argv)


@pytest.mark.parametrize(
    "argv, count",
    [
        (["--", "--port", "1"], 2),
        (["--port", "2", "--", "--tokn=s3cret", "-s3cret"], 2),
        (["--", "--"], 1),
        (["--nope", "--", "s3cret"], 1),
    ],
)
def test_every_token_after_a_bare_double_dash_is_a_positional(argv, count) -> None:
    with pytest.raises(ValueError) as caught:
        load(argv=argv)
    message = str(caught.value)
    assert (
        f"  the arguments (arg): no item takes a positional argument, got {count}; "
        "a server passes argv=sys.argv[1:]" in message
    )
    assert "s3cret" not in message and "--port" not in message
    assert ("no item takes this flag" in message) == ("--nope" in argv)


def test_a_bare_double_dash_alone_is_no_positional() -> None:
    assert load(argv=["--port", "2", "--"]).config.port == 2


def test_a_positional_says_to_pass_argv_from_one() -> None:
    with pytest.raises(ValueError) as caught:
        load(argv=["/usr/bin/server", "--port", "1"])
    message = str(caught.value)
    assert (
        "  the arguments (arg): no item takes a positional argument, got 1; a server passes argv=sys.argv[1:]"
        in message
    )
    assert "/usr/bin/server" not in message


def test_a_flag_without_its_value_is_a_collected_error_not_an_exit() -> None:
    with pytest.raises(
        ValueError,
        match=r"  the arguments \(arg\): argument --port: expected one argument",
    ):
        load(argv=["--port"])


def test_a_flag_without_its_value_reads_no_file_and_still_reports_env(
    tmp_path,
) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("nope: 1\n")
    with pytest.raises(ValueError) as caught:
        load(environ={"CONFIG_FILE": str(f), "PORT": "x"}, argv=["--config-file"])
    message = str(caught.value)
    assert "  the arguments (arg): argument --config-file: expected one argument" in (
        message
    )
    assert "  port (env PORT): Input should be a valid integer" in message
    assert "nope" not in message


@pytest.mark.parametrize("argv", ["--port 1", b"--port", ["--port", 1]])
def test_argv_must_be_a_list_of_strings(argv) -> None:
    with pytest.raises(TypeError, match=r"sys\.argv\[1:\]"):
        load(argv=argv)


def test_flags_that_differ_only_in_dashes_and_underscores_both_load() -> None:
    loaded = load(Dashes, argv=["--a-b", "1", "--a_b", "2"])
    assert (loaded.config.dashes.a, loaded.config.dashes.b) == (1, 2)


def test_a_repeated_flag_keeps_its_last_value() -> None:
    assert load(argv=["--port", "1", "--port", "2"]).config.port == 2


# -- values


@config("jsoned")
class Jsoned(Config):
    ids: Json[list[int]] = Field(default_factory=list, env="JSONED_IDS")


def test_scalars_are_strings_validated_in_lax_mode() -> None:
    assert load(environ={"PORT": "9000"}).config.port == 9000


def test_lists_dicts_and_models_take_json_from_env_and_flags() -> None:
    loaded = load(
        environ={"RECORDING_HOSTS": '["a", "b"]', "TELEMETRY_LIMITS": '{"burst": 3}'},
        argv=["--recording-tags", '{"k": "v"}'],
    )
    assert loaded.config.recording.hosts == ["a", "b"]
    assert loaded.config.recording.tags == {"k": "v"}
    assert loaded.config.telemetry.limits.burst == 3


@pytest.mark.parametrize(
    "environ, line",
    [
        (
            {"RECORDING_TAGS": '{"s3cret"'},
            "  recording.tags (env RECORDING_TAGS): must be a JSON object",
        ),
        (
            {"RECORDING_TAGS": '["s3cret"]'},
            "  recording.tags (env RECORDING_TAGS): must be a JSON object",
        ),
        (
            {"RECORDING_HOSTS": '{"s3cret": 1}'},
            "  recording.hosts (env RECORDING_HOSTS): must be a JSON array",
        ),
    ],
)
def test_bad_json_names_the_item_and_source_never_the_value(environ, line) -> None:
    with pytest.raises(ValueError) as caught:
        load(environ=environ)
    assert line in str(caught.value) and "s3cret" not in str(caught.value)


def test_no_decode_passes_the_raw_string() -> None:
    assert load(environ={"RECORDING_RAW": "a,b"}).config.recording.raw == ["a", "b"]


def test_a_json_field_takes_its_raw_string() -> None:
    assert paths(Jsoned)["jsoned.ids"].takes == "string"
    assert load(Jsoned, environ={"JSONED_IDS": "[1, 2]"}).config.jsoned.ids == [1, 2]


# -- errors


def test_errors_from_every_source_are_reported_together(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("recording:\n  nope: 1\n")
    with pytest.raises(ValueError) as caught:
        load(
            environ={"CONFIG_FILE": str(f), "PORT": "x"},
            argv=["--recording-tags", "[1]"],
        )
    message = str(caught.value)
    assert message.startswith("the config did not load:\n")
    assert "  recording.tags (arg --recording-tags): must be a JSON object" in message
    assert "  port (env PORT): Input should be a valid integer" in message
    assert f"  recording.nope (file {f}): Extra inputs are not permitted" in message
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_mismatch_is_the_highest_layer_holding_it(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("port: 1\n")
    with pytest.raises(ValueError, match=r"  port \(env PORT\): "):
        load(environ={"CONFIG_FILE": str(f), "PORT": "x"})


# -- provenance


def test_every_source_is_recorded(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("name: kb\ntelemetry:\n  limits:\n    burst: 7\n")
    loaded = load(
        environ={"CONFIG_FILE": str(f), "TELEMETRY_RATE": "9"}, argv=["--port", "1"]
    )
    assert loaded.sources["port"] == Source("arg", "--port")
    assert loaded.sources["name"] == Source("file", str(f))
    assert loaded.sources["telemetry.limits.burst"] == Source("file", str(f))
    assert loaded.sources["telemetry.limits.rate"] == Source("env", "TELEMETRY_RATE")
    assert loaded.sources["recording.inbox"] == Source("default")


def test_a_blank_leaves_the_lower_source(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("port: 8001\n")
    loaded = load(environ={"CONFIG_FILE": str(f), "PORT": ""}, argv=["--port", " "])
    assert loaded.sources["port"] == Source("file", str(f))


def test_the_record_covers_every_leaf_and_is_read_only() -> None:
    sources = load().sources
    assert list(sources) == [
        i.path for i in _compose([[Serve, Rec, Tel]]).items if i.leaf
    ]
    assert set(sources.values()) == {Source("default")}
    with pytest.raises(TypeError):
        sources["port"] = Source("code")


def test_values_in_code_are_code() -> None:
    sources = _load([[Serve, Rec]], {"port": 2, "recording": Rec(inbox="/in")}).sources
    assert sources["port"] == Source("code")
    assert sources["recording.hosts"] == Source("code")
    assert sources["name"] == Source("default")


def test_a_model_field_set_from_json_names_its_env_var() -> None:
    sources = load(environ={"TELEMETRY_LIMITS": '{"burst": 3}'}).sources
    assert sources["telemetry.limits.burst"] == Source("env", "TELEMETRY_LIMITS")
    assert sources["telemetry.limits.rate"] == Source("default")


def test_a_source_prints_as_it_appears_in_an_error() -> None:
    sources = (
        Source("default"),
        Source("code"),
        Source("env", "PORT"),
        Source("arg", "--port"),
        Source("file", "/etc/c.yaml"),
        Source("hook"),
    )
    assert [str(s) for s in sources] == [
        "default",
        "code",
        "env PORT",
        "arg --port",
        "file /etc/c.yaml",
        "hook",
    ]


def test_the_boot_line_lists_only_what_is_not_default(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="kubed.mcp.config"):
        load(
            environ={"PORT": "9000", "RECORDING_TOKEN": "s3cret"},
            argv=["--recording-inbox", "/in"],
        )
        load()
    lines = [r.getMessage() for r in caplog.records if r.name == "kubed.mcp.config"]
    assert lines == [
        'config: port=9000 (env PORT), recording.inbox="/in" (arg --recording-inbox), '
        "recording.token=*** (env RECORDING_TOKEN)",
        "config: all defaults",
    ]


def test_the_boot_line_is_not_built_when_info_is_off(caplog, monkeypatch) -> None:
    def refuse(*args: object) -> str:
        raise AssertionError("the boot line was built")

    monkeypatch.setattr("kubed.mcp._config._boot_line", refuse)
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.config"):
        assert load(environ={"PORT": "9000"}).config.port == 9000


def test_a_load_that_fails_logs_no_boot_line(caplog) -> None:
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
        load(environ={"PORT": "x"})
    assert not [r for r in caplog.records if r.name == "kubed.mcp.config"]


@config("loose")
class Loose(Config):
    thing: Any = None
    blob: bytes = b""


def test_a_value_json_cannot_write_is_named_by_its_type(caplog) -> None:
    values = {"loose": {"thing": SimpleNamespace(t="s3cret"), "blob": b"\xffs3cret"}}
    with caplog.at_level(logging.INFO, logger="kubed.mcp.config"):
        _load([[Loose]], values)
    assert [r.getMessage() for r in caplog.records if r.name == "kubed.mcp.config"] == [
        'config: loose.thing="<SimpleNamespace>" (code), loose.blob="<bytes>" (code)'
    ]


class Login(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    token: SecretStr

    @field_serializer("token", when_used="json")
    def _reveal(self, token: SecretStr) -> str:
        return token.get_secret_value()


class Keyring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creds: list[Login] = []


@config("logins")
class Logins(Config):
    creds: list[Login] = Field(default_factory=list, env="LOGINS_CREDS")
    deep: dict[str, Keyring] = Field(default_factory=dict, env="LOGINS_DEEP")
    one: Login | None = Field(None)


def test_the_boot_line_masks_a_leaf_that_holds_a_secret_at_any_depth(caplog) -> None:
    environ = {
        "LOGINS_CREDS": '[{"name": "a", "token": "S3CRET-LIST"}]',
        "LOGINS_DEEP": '{"x": {"creds": [{"name": "b", "token": "S3CRET-DEEP"}]}}',
    }
    with caplog.at_level(logging.INFO, logger="kubed.mcp.config"):
        loaded = load(Logins, environ=environ).config
    assert loaded.logins.creds[0].token.get_secret_value() == "S3CRET-LIST"
    assert "S3CRET" not in caplog.text
    assert [r.getMessage() for r in caplog.records if r.name == "kubed.mcp.config"] == [
        "config: logins.creds=*** (env LOGINS_CREDS), logins.deep=*** (env LOGINS_DEEP)"
    ]


needs_type_params = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="typing.TypeAliasType is 3.12+"
)


def _generic_aliases() -> tuple[Any, Any]:
    """``Many[T] = list[T]`` and ``Pair[K, V] = dict[K, V]``, as TypeAliasTypes."""
    import typing

    t, k, v = typing.TypeVar("T"), typing.TypeVar("K"), typing.TypeVar("V")
    many = typing.TypeAliasType("Many", list[t], type_params=(t,))
    pair = typing.TypeAliasType("Pair", dict[k, v], type_params=(k, v))
    return many, pair


@needs_type_params
def test_the_boot_line_masks_a_generic_alias_that_holds_a_secret(caplog) -> None:
    many, pair = _generic_aliases()

    @config("aliased")
    class Aliased(Config):
        creds: many[Login] = Field(default_factory=list, env="ALIASED_CREDS")
        named: pair[str, Login] = Field(default_factory=dict, env="ALIASED_NAMED")

    environ = {
        "ALIASED_CREDS": '[{"name": "a", "token": "S3CRET-MANY"}]',
        "ALIASED_NAMED": '{"x": {"name": "b", "token": "S3CRET-PAIR"}}',
    }
    with caplog.at_level(logging.INFO, logger="kubed.mcp.config"):
        loaded = load(Aliased, environ=environ).config
    assert loaded.aliased.named["x"].token.get_secret_value() == "S3CRET-PAIR"
    assert "S3CRET" not in caplog.text
    assert [r.getMessage() for r in caplog.records if r.name == "kubed.mcp.config"] == [
        "config: aliased.creds=*** (env ALIASED_CREDS), "
        "aliased.named=*** (env ALIASED_NAMED)"
    ]


@needs_type_params
def test_a_bad_key_under_a_generic_alias_of_secrets_never_prints() -> None:
    _, pair = _generic_aliases()

    @config("aliaskeys")
    class AliasKeys(Config):
        keys: pair[str, SecretStr] = Field(default_factory=dict, env="ALIASKEYS_KEYS")

    with pytest.raises(ValueError) as caught:
        load(AliasKeys, environ={"ALIASKEYS_KEYS": '{"s3cretkey": 5}'})
    message = str(caught.value)
    assert "  aliaskeys.keys (env ALIASKEYS_KEYS): " in message
    assert "s3cretkey" not in message


@needs_type_params
def test_a_model_under_a_generic_alias_must_forbid_unknown_keys() -> None:
    many, _ = _generic_aliases()

    class Loose(BaseModel):
        name: str = ""

    @config("aliasloose")
    class AliasLoose(Config):
        items: many[Loose] = Field(default_factory=list)

    with pytest.raises(TypeError, match=r"Loose at aliasloose\.items allows unknown"):
        _compose([[AliasLoose]])


# -- secrets


@config("pair")
class Pair(Config):
    a: int = Field(0, env="PAIR_A")
    b: int = Field(0, env="PAIR_B")

    @model_validator(mode="after")
    def _same(self) -> Pair:
        if self.a != self.b:
            raise ValueError("a and b differ")
        return self


class Cred(Config):
    keys: Secret[dict[str, int]]


@config("vault")
class Vault(Config):
    creds: list[Cred] = Field(default_factory=list)


class Plain(Config):
    note: str = ""


class Locked(Config):
    keys: Secret[dict[str, int]] | None = None


@config("either")
class Either(Config):
    one: Plain | Locked = Plain()


class Entry(Config):
    a: int = 0
    s: Secret[dict[str, int]] | None = None


@config("d")
class Entries(Config):
    d: dict[str, Entry] = Field(default_factory=dict)


def test_a_secret_from_env_never_shows(caplog) -> None:
    environ = {"RECORDING_TOKEN": "s3cret"}
    with caplog.at_level(logging.DEBUG):
        loaded = load(environ=environ)
    for text in (
        caplog.text,
        repr(loaded.sources),
        repr(loaded.config),
        repr(loaded.config.recording),
        loaded.config.model_dump_json(),
        loaded.config.recording.model_dump_json(),
    ):
        assert "s3cret" not in text
    with pytest.raises(ValueError) as caught:
        load(environ={**environ, "PORT": "x"})
    assert "s3cret" not in str(caught.value)


def test_a_bad_key_of_a_secret_dict_never_prints() -> None:
    with pytest.raises(ValueError) as caught:
        load(environ={"RECORDING_KEYS": '{"s3cretkey": "x"}'})
    message = str(caught.value)
    assert "  recording.keys (env RECORDING_KEYS): " in message
    assert "s3cretkey" not in message


def test_the_path_stops_at_a_secret_in_a_list_of_models() -> None:
    with pytest.raises(ValueError) as caught:
        _load([[Vault]], {"vault": {"creds": [{"keys": {"s3cretkey": "x"}}]}})
    message = str(caught.value)
    assert "  vault.creds[0].keys (code): " in message
    assert "s3cretkey" not in message


def test_the_path_stops_at_a_secret_in_a_union_of_models() -> None:
    with pytest.raises(ValueError) as caught:
        _load([[Either]], {"either": {"one": {"keys": {"s3cretkey": "x"}}}})
    message = str(caught.value)
    assert "either.one.Locked.keys (" in message and "s3cretkey" not in message


def test_a_dict_key_named_like_a_field_still_stops_at_a_secret() -> None:
    # The dict key "a" is also Entry's field a, which must not hide Entry's secret s.
    with pytest.raises(ValueError) as caught:
        _load([[Entries]], {"d": {"d": {"a": {"s": {"s3cretkey": "x"}}}}})
    message = str(caught.value)
    assert "  d.d.a.s (code): " in message and "s3cretkey" not in message


@config("byname")
class ByName(Config):
    model_config = ConfigDict(validate_by_name=True)
    keys: Secret[dict[str, int]] | None = Field(None, validation_alias="Keys")


@config("populated")
class Populated(Config):
    model_config = ConfigDict(populate_by_name=True)
    keys: Secret[dict[str, int]] | None = Field(None, validation_alias="Keys")


@pytest.mark.parametrize(
    "section, values, path",
    [
        (Routes, {"routes": {"k": {"s3cretkey": "x"}}}, "routes.k"),
        (Routes, {"routes": {"keys": {"s3cretkey": "x"}}}, "routes.keys"),
        (Routes, {"routes": {"bb": [{"s3cretkey": "x"}]}}, "routes.bb"),
        (Routes, {"routes": {"z": {"s3cretkey": "x"}}}, "routes.z"),
        (Routes, {"routes": {"cc": [{"s3cretkey": "x"}]}}, "routes.cc"),
        (ByName, {"byname": {"keys": {"s3cretkey": "x"}}}, "byname.keys"),
        (ByName, {"byname": {"Keys": {"s3cretkey": "x"}}}, "byname.Keys"),
        (Populated, {"populated": {"keys": {"s3cretkey": "x"}}}, "populated.keys"),
    ],
)
def test_the_path_stops_at_a_secret_under_any_key_it_accepts(
    section, values, path
) -> None:
    with pytest.raises(ValueError) as caught:
        _load([[section]], values)
    message = str(caught.value)
    assert f"\n  {path} (code): " in message and "s3cretkey" not in message


class Knobs(Config):
    burst: int = Field(1, validation_alias=AliasChoices("b", "bb"))
    rate: int = Field(2, env="KNOBS_RATE")


@config("knobs")
class KnobsSection(Config):
    knobs: Knobs = Knobs(b=5)


def test_a_default_instance_merges_under_the_key_its_field_accepts(tmp_path) -> None:
    knobs = load(KnobsSection, environ={"KNOBS_RATE": "9"}).config.knobs.knobs
    assert (knobs.burst, knobs.rate) == (5, 9)
    f = tmp_path / "c.yaml"
    f.write_text("knobs:\n  knobs:\n    bb: 7\n")
    knobs = load(KnobsSection, environ={"CONFIG_FILE": str(f)}).config.knobs.knobs
    assert (knobs.burst, knobs.rate) == (7, 2)


def test_a_section_error_with_several_names_below_it_names_the_kind_only() -> None:
    with pytest.raises(ValueError, match=r"  pair \(env\): .*a and b differ"):
        load(Pair, environ={"PAIR_A": "1", "PAIR_B": "2"})


# -- a server's sources


def test_a_server_reads_only_the_flags_and_env_it_is_given(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "1")
    mcp = QuickMCP("t", sections=[Serve], argv=["--port", "2"], environ={})
    assert mcp.config.port == 2
    assert mcp.config_sources["port"] == Source("arg", "--port")
    assert QuickMCP("t", sections=[Serve], environ={"PORT": "3"}).config.port == 3


def test_a_server_built_without_environ_reads_the_process_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "4")
    mcp = QuickMCP("t", sections=[Serve])
    assert mcp.config_sources["port"] == Source("env", "PORT")


def test_config_with_argv_or_environ_is_a_type_error_through_the_server() -> None:
    with pytest.raises(TypeError, match="config= reads no flags or env vars"):
        QuickMCP("t", sections=[Serve], config={"port": 1}, argv=["--port", "2"])


def test_a_bad_flag_stops_construction_without_exiting() -> None:
    with pytest.raises(ValueError, match=r"the arguments \(arg --nope\)"):
        QuickMCP("t", sections=[Serve], argv=["--nope"], environ={})


def test_config_sources_is_read_only() -> None:
    mcp = QuickMCP("t", sections=[Serve], environ={})
    with pytest.raises(AttributeError):
        mcp.config_sources = {}
    with pytest.raises(TypeError):
        mcp.config_sources["port"] = Source("code")


def test_the_public_names_are_the_modules_own() -> None:
    import kubed.mcp

    assert kubed.mcp.Source is Source and kubed.mcp.NoDecode is NoDecode
