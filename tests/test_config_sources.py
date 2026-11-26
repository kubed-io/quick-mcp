"""Where a config's values come from: its items, their names, and its secrets."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Annotated, Optional

import pytest
from pydantic import Json, Secret, SecretStr, create_model, field_validator

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
