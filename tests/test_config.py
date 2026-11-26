"""The config core: sections by key, composed into one strict config model."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import types
import uuid
import zoneinfo
from types import MappingProxyType
from typing import Annotated, Any, Literal, NewType

import pytest
from fastmcp import Client
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    ImportString,
    Json,
    RootModel,
    SecretStr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import TypeAliasType, TypedDict

import kubed.mcp._config
from kubed.mcp import Inject, QuickMCP
from kubed.mcp._config import Config, Field, _compose, _load, config, config_schema

pytestmark = pytest.mark.unit


# -- sections


@config()
class Root(Config):
    name: str = "flow"

    @property
    def shout(self) -> str:
        return self.name.upper()


@config("recording")
class Recording(Config):
    inbox: str | None = Field(
        None, doc="Where recordings go", env="RECORDING_INBOX", arg="--recording-inbox"
    )

    @property
    def enabled(self) -> bool:
        return self.inbox is not None


class Unstamped(Recording):
    pass


class Deep(Config):
    level: int = 3

    @property
    def twice(self) -> int:
        return self.level * 2


class Middle(Config):
    deep: Deep = Field(default_factory=Deep)


@config("nest")
class Nest(Config):
    label: str = ""
    middle: Middle = Field(default_factory=Middle)

    @field_validator("label")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _name_it(self) -> Nest:
        if not self.label:
            self.label = "unnamed"
        return self


@config("base")
class Base(Config):
    a: int = 1


@config("base")
class Sub(Base):
    b: int = 2


@config("base")
class Other(Config):
    c: int = 3


class Node(Config):
    children: list[Node] = Field(default_factory=list)


@config("tree")
class HoldsNode(Config):
    root: Node = Node()


@config()
class Clashing(Config):
    recording: str = ""


@config()
class Shouting(Config):
    @property
    def loud(self) -> str:
        return "LOUD"


@config("loud")
class Loud(Config):
    pass


@config()
class Methodical(Config):
    def run(self) -> None:
        pass


@config("run")
class Runs(Config):
    pass


class Loose(BaseModel):
    x: int = 0


@config("recording2")
class Holding(Config):
    inner: Loose | None = None


@config("listed")
class HoldsList(Config):
    items: list[Loose] = Field(default_factory=list)


@config("mapped")
class HoldsDict(Config):
    items: dict[str, Loose] = Field(default_factory=dict)


@config("wrapped")
class HoldsAnnotated(Config):
    item: Annotated[Loose, "meta"] | None = None


@config("loose")
class LooseSection(Loose):
    pass


@dataclasses.dataclass
class Point:
    x: int = 0


@config("pointy")
class HoldsPoint(Config):
    point: Point | None = None


class Typed(TypedDict):
    x: int


@config("typed")
class HoldsTyped(Config):
    value: Typed | None = None


@config("needs")
class Needy(Config):
    required: int


class Secret(Config):
    token: str = "hunter2"


@config("early")
class HoldsLater(Config):
    later: Later | None = None


class Later(BaseModel):
    x: int = 0


@config("never")
class HoldsMissing(Config):
    gone: NeverDefined | None = None  # noqa: F821


WrappedLoose = NewType("WrappedLoose", Loose)
AliasedLoose = TypeAliasType("AliasedLoose", Loose)


@config("newtyped")
class HoldsNewType(Config):
    item: WrappedLoose | None = None


@config("aliased")
class HoldsAlias(Config):
    item: AliasedLoose | None = None


@config()
class UnfrozenRoot(Config):
    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str = "x"


@pytest.mark.usefixtures("no_builtins")
def test_config_stamps_the_key_and_returns_the_class() -> None:
    class Fresh(Config):
        pass

    assert config("fresh")(Fresh) is Fresh
    assert _compose([[Fresh]]).held == {"fresh": Fresh}
    # a subclass with no @config of its own doesn't inherit its parent's key
    with pytest.raises(TypeError, match=r"Unstamped has no @config"):
        _compose([[Unstamped]])


@pytest.mark.parametrize(
    "key", ["", "1st", "_private", "model_x", "class", "json", "copy"]
)
def test_a_bad_key_is_a_value_error(key) -> None:
    with pytest.raises(ValueError, match=repr(key)):
        config(key)


def test_a_key_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        config(3)


def test_config_refuses_a_class_that_is_not_a_model_and_a_bare_decorator() -> None:
    with pytest.raises(TypeError, match="BaseModel"):
        config("x")(type("Plain", (), {}))
    with pytest.raises(TypeError, match=r"@config\(\)"):
        config(Recording)  # @config with no parentheses


def test_field_writes_doc_env_arg_and_sensitive() -> None:
    info = Field("x", doc="A doc", env="A_ENV", arg="--a", sensitive=True)
    assert info.description == "A doc"
    assert info.json_schema_extra == {"env": "A_ENV", "arg": "--a", "sensitive": True}
    assert info.repr is True  # a secret type masks itself, so repr is pydantic's
    bare = Field(doc="d")
    assert bare.json_schema_extra is None  # no item given, nothing written
    assert bare.is_required()  # no default given: required, as pydantic's Field
    assert Field("x", env="E").json_schema_extra == {"env": "E"}
    with pytest.raises(TypeError, match="doc="):
        Field("x", description="d")
    with pytest.raises(TypeError, match="doc="):
        Field("x", json_schema_extra={})
    with pytest.raises(TypeError, match=r"env=\) must be a non-empty str"):
        Field("x", env="")
    with pytest.raises(TypeError, match=r"arg=\) must be a non-empty str"):
        Field("x", arg=3)


def test_field_passes_other_keywords_to_pydantic() -> None:
    class Bounded(BaseModel):
        n: int = Field(3, ge=1)

    assert Bounded().n == 3
    with pytest.raises(ValidationError):
        Bounded(n=0)


# -- composing


@pytest.mark.usefixtures("no_builtins")
def test_a_root_and_keyed_sections_compose() -> None:
    composed = _compose([[Root, Recording]])
    assert set(composed.model.model_fields) == {"name", "recording"}
    assert issubclass(composed.model, Root) and issubclass(composed.model, Config)
    assert composed.held == {None: Root, "recording": Recording}
    loaded = composed.model()
    assert loaded.shout == "FLOW"
    assert loaded.recording.enabled is False
    assert loaded.recording.inbox is None


def test_the_composed_top_is_frozen_and_forbids_unknown_keys() -> None:
    model = _compose([[Root, Recording]]).model
    loaded = model()
    with pytest.raises(ValueError, match="frozen"):
        loaded.name = "x"
    with pytest.raises(ValueError, match="Extra"):
        model(nope=1)


def test_sections_are_not_frozen_and_their_validators_work() -> None:
    model = _compose([[Nest]]).model
    loaded = model(nest={"label": "  hi  "})
    assert loaded.nest.label == "hi"
    assert model().nest.label == "unnamed"
    assert model().nest.middle.deep.twice == 6


def test_a_section_with_a_required_field_is_required() -> None:
    model = _compose([[Needy]]).model
    assert model.model_fields["needs"].is_required()
    assert not _compose([[Recording]]).model.model_fields["recording"].is_required()


@pytest.mark.usefixtures("no_builtins")
def test_a_composition_without_a_root_has_only_keyed_fields() -> None:
    composed = _compose([[Recording]])
    assert set(composed.model.model_fields) == {"recording"}
    assert None not in composed.held


def test_an_unrelated_class_under_a_taken_key_names_both() -> None:
    with pytest.raises(ValueError, match=r"'base'.*Base.*Other|'base'.*Other.*Base"):
        _compose([[Base, Other]])


def test_two_unrelated_roots_name_the_root_section() -> None:
    @config()
    class OtherRoot(Config):
        pass

    with pytest.raises(ValueError, match=r"root section.*Root.*OtherRoot"):
        _compose([[Root, OtherRoot]])


@pytest.mark.parametrize("order", [[Base, Sub], [Sub, Base]])
def test_a_subclass_replaces_its_parent_in_either_order(order) -> None:
    composed = _compose([order])
    assert composed.held["base"] is Sub
    assert composed.replaced["base"] == (Base,)


def test_the_same_class_again_is_a_no_op() -> None:
    composed = _compose([[Base], [Base, Sub, Sub, Base]])
    assert composed.held["base"] is Sub
    assert composed.replaced["base"] == (Base,)


def test_a_servers_subclass_replaces_a_built_in(monkeypatch) -> None:
    monkeypatch.setattr(kubed.mcp._config, "_builtin_sections", lambda: (Base,))
    composed = _compose([[Sub]])
    assert composed.held["base"] is Sub
    assert composed.replaced["base"] == (Base,)


def test_a_root_field_named_like_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'recording'.*Clashing.*Recording"):
        _compose([[Clashing, Recording]])


def test_a_key_that_shadows_an_attribute_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'loud'.*Shouting.*Loud"):
        _compose([[Shouting, Loud]])
    with pytest.raises(ValueError, match=r"'run'.*Methodical.*Runs"):
        _compose([[Methodical, Runs]])


def test_a_class_or_an_instance_where_sources_belong_is_a_type_error() -> None:
    with pytest.raises(TypeError, match=r"sections=\(Root,\)"):
        _compose([Root])
    with pytest.raises(TypeError, match="config="):
        _compose([[Root()]])


@pytest.mark.parametrize(
    ("section", "path"),
    [
        (Holding, r"recording2\.inner"),
        (HoldsList, r"listed\.items"),
        (HoldsDict, r"mapped\.items"),
        (HoldsAnnotated, r"wrapped\.item"),
        (LooseSection, r"loose"),
    ],
)
def test_a_nested_model_that_allows_unknown_keys_is_refused(section, path) -> None:
    with pytest.raises(TypeError, match=rf"Loose\w* at {path}"):
        _compose([[section]])


def test_a_loose_root_is_named_the_root_section() -> None:
    @config()
    class LooseRoot(BaseModel):
        x: int = 0

    with pytest.raises(TypeError, match="LooseRoot at the root section"):
        _compose([[LooseRoot]])


def test_a_loose_model_in_a_root_field_is_named_by_the_field() -> None:
    @config()
    class HoldingRoot(Config):
        inner: Loose | None = None

    with pytest.raises(TypeError, match=r"Loose at inner"):
        _compose([[HoldingRoot]])


def test_a_nested_dataclass_or_typed_dict_is_refused() -> None:
    with pytest.raises(TypeError, match=r"Point at pointy\.point"):
        _compose([[HoldsPoint]])
    with pytest.raises(TypeError, match=r"Typed at typed\.value"):
        _compose([[HoldsTyped]])


def test_a_self_referencing_model_registers() -> None:
    composed = _compose([[HoldsNode]])
    assert composed.model().tree.root.children == []


def test_a_roots_own_frozen_false_does_not_unfreeze_the_top() -> None:
    loaded = _compose([[UnfrozenRoot]]).model()
    with pytest.raises(ValidationError):
        loaded.name = "y"


def test_no_message_holds_a_config_value() -> None:
    secret = Secret()
    attempts = [
        lambda: _compose([[secret]]),
        lambda: _compose([secret]),
        lambda: config("x")(secret),
        lambda: config(secret),
        lambda: Field("x", env=secret),
    ]
    for attempt in attempts:
        with pytest.raises(TypeError) as caught:
            attempt()
        assert "hunter2" not in str(caught.value)
    with pytest.raises(TypeError, match=r"instance of Secret.*config="):
        _compose([secret])


def test_a_forward_reference_to_a_loose_model_is_refused() -> None:
    with pytest.raises(TypeError, match=r"Later at early\.later"):
        _compose([[HoldsLater]])


def test_a_reference_that_never_resolves_is_a_type_error() -> None:
    with pytest.raises(TypeError, match=r"HoldsMissing at never"):
        _compose([[HoldsMissing]])


def test_a_new_type_and_a_type_alias_are_followed() -> None:
    with pytest.raises(TypeError, match=r"Loose at newtyped\.item"):
        _compose([[HoldsNewType]])
    with pytest.raises(TypeError, match=r"Loose at aliased\.item"):
        _compose([[HoldsAlias]])


# -- loading


@config("required")
class Required(Config):
    token: str


@config("secretive")
class Secretive(Config):
    token: SecretStr = Field(SecretStr("dflt"), sensitive=True)


class Unregistered(Config):
    token: str = "hunter2"


@config("node")
class HoldsTree(Config):
    children: list[Node] = Field(default_factory=list)


@config()
class Failing(Config):
    @model_validator(mode="after")
    def _never(self) -> Failing:
        raise ValueError("no way")


class TagA(Config):
    kind: Literal["a"]


class TagB(Config):
    kind: Literal["b"]


@config("leaky")
class Leaky(Config):
    tag: Annotated[TagA | TagB, Field(discriminator="kind")] | None = None
    module: ImportString | None = None
    ident: uuid.UUID | None = None
    when: datetime.datetime | None = None
    doc: Json[int] | None = None
    zone: zoneinfo.ZoneInfo | None = None
    mail: EmailStr | None = None


@config("keyed")
class Keyed(Config):
    m: dict[int, int] = Field(default_factory=dict)


@config("keyfail")
class KeyFail(Config):
    @model_validator(mode="after")
    def _refuse(self) -> KeyFail:
        raise ValueError("nope keyed")


class BadInner(Config):
    x: int = Field("bad", validate_default=True)


@config("outer")
class Outer(Config):
    inner: BadInner = Field(default_factory=dict, validate_default=True)


@config()
class R1(Config):
    pass


@config()
class R2(R1):
    pass


@config()
class RootOf(Config):
    pass


@config("x")
class XOfRoot(RootOf):
    pass


class Child(XOfRoot):
    pass


@dataclasses.dataclass
class Settings:
    port: int = 8000


@pytest.mark.usefixtures("no_builtins")
def test_no_sections_load_an_empty_config() -> None:
    loaded = _load([[]], None, environ={})
    assert type(loaded.config).model_fields == {}
    assert isinstance(loaded.config, Config)
    assert loaded.provided == {Config: loaded.config}


def test_defaults_load_and_properties_and_validators_run() -> None:
    loaded = _load([[Root, Recording, Nest]], None, environ={})
    assert loaded.config.name == "flow" and loaded.config.shout == "FLOW"
    assert loaded.config.recording.enabled is False
    # a section's assigning validator still works
    assert loaded.config.nest.label == "unnamed"


def test_three_levels_load_from_a_dict() -> None:
    loaded = _load(
        [[Nest]], {"nest": {"label": " x ", "middle": {"deep": {"level": 4}}}}
    )
    assert loaded.config.nest.label == "x"
    assert loaded.config.nest.middle.deep.twice == 8


def test_a_dict_may_hold_section_instances() -> None:
    recording = Recording(inbox="/in")
    loaded = _load([[Root, Recording]], {"name": "kb", "recording": recording})
    assert loaded.config.recording is recording and loaded.config.name == "kb"


def test_section_instances_go_to_their_keys_and_a_root_spreads() -> None:
    recording = Recording(inbox="/in")
    loaded = _load([[Root, Recording]], [Root(name="kb"), recording])
    assert loaded.config.name == "kb" and loaded.config.recording is recording


def test_a_single_instance_is_one_section() -> None:
    assert _load([[Root]], Root(name="kb")).config.name == "kb"


def test_a_non_dict_mapping_is_accepted() -> None:
    assert _load([[Root]], MappingProxyType({"name": "kb"})).config.name == "kb"


def test_the_provided_values_cover_every_class() -> None:
    loaded = _load([[Root, Base, Sub]], None, environ={})
    assert loaded.provided[Sub] is loaded.provided[Base] is loaded.config.base
    assert loaded.provided[Config] is loaded.provided[Root] is loaded.config


@pytest.mark.parametrize(
    "values, path",
    [
        ({"nope": 1}, "nope (code)"),
        ({"recording": {"nope": 1}}, "recording.nope (code)"),
        ({"nest": {"middle": {"deep": {"nope": 1}}}}, "nest.middle.deep.nope (code)"),
    ],
)
def test_an_unknown_key_names_its_path_and_source(values, path) -> None:
    with pytest.raises(ValueError, match=r"^the config did not load:") as caught:
        _load([[Root, Recording, Nest]], values)
    assert f"  {path}: Extra inputs are not permitted" in str(caught.value)


def test_a_missing_required_section_is_from_default() -> None:
    with pytest.raises(ValueError, match=r"required \(default\): Field required"):
        _load([[Required]], None, environ={})


def test_a_list_index_is_written_in_brackets() -> None:
    with pytest.raises(ValueError, match=r"  node\.children\[0\]\.nope \(code\)"):
        _load([[HoldsTree]], {"node": {"children": [{"nope": 1}]}})


def test_an_email_error_never_holds_the_input() -> None:
    for mail in ("a@s3cr\u00e9t\u0661\u06f1.com", "a@s3cr_t.com"):
        with pytest.raises(ValueError) as caught:
            _load([[Leaky]], {"leaky": {"mail": mail}})
        message = str(caught.value)
        assert "s3cr" not in message
        assert "leaky.mail (code): value is not a valid email address" in message


def test_a_dict_key_error_has_no_stray_dot() -> None:
    with pytest.raises(ValueError, match=r"  keyed\.m\.k\[key\] \(code\)"):
        _load([[Keyed]], {"keyed": {"m": {"k": 1}}})


def test_an_error_with_no_path_names_the_root_section() -> None:
    with pytest.raises(ValueError, match=r"  the root section \(default\): .*no way"):
        _load([[Failing]], None, environ={})


def test_a_load_error_never_holds_a_value() -> None:
    with pytest.raises(ValueError) as caught:
        _load([[Secretive]], {"secretive": {"token": ["s3cret"]}})
    assert "s3cret" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize(
    "values, match",
    [
        (Settings(), "sections="),  # a dataclass, as test_app's
        (argparse.Namespace(port=1), "sections="),
        (Unregistered(), "sections="),  # a Config subclass never registered
        ([Recording], "sections="),  # a class among values
        ("recording", "mapping or section instances"),
        (b"recording", "mapping or section instances"),
        (42, "mapping or section instances"),
        (Recording, "the class Recording"),
    ],
)
def test_an_old_style_config_is_a_type_error_naming_sections(values, match) -> None:
    with pytest.raises(TypeError, match=match) as caught:
        _load([[Root, Recording]], values)
    assert "sections=" in str(caught.value)


def test_an_unregistered_instance_never_leaks_its_values() -> None:
    for values in (Unregistered(token="s3cret"), [Unregistered(token="s3cret")]):
        with pytest.raises(TypeError) as caught:
            _load([[Root]], values)
        assert "s3cret" not in str(caught.value)
        assert "Unregistered" in str(caught.value)


def test_an_instance_of_a_replaced_class_names_the_winner() -> None:
    with pytest.raises(TypeError, match="Sub"):
        _load([[Base, Sub]], [Base()])


def test_two_instances_for_one_key_are_refused() -> None:
    with pytest.raises(ValueError, match="'recording'"):
        _load([[Recording]], [Recording(), Recording()])


def test_instances_are_kept_as_given() -> None:
    recording = Recording(inbox="/in")
    assert _load([[Recording]], {"recording": recording}).config.recording is recording
    assert _load([[Recording]], [recording]).config.recording is recording


def test_a_source_is_code_only_for_a_key_that_was_given() -> None:
    with pytest.raises(ValueError) as caught:
        _load([[Required, Recording]], {"recording": {"nope": 1}})
    message = str(caught.value)
    assert "recording.nope (code)" in message
    assert "required (default): Field required" in message


_SECRET = "s3cret"


@pytest.mark.parametrize(
    "values",
    [
        {"tag": {"kind": _SECRET}},
        {"module": f"{_SECRET}.mod"},
        {"ident": f"{_SECRET}-zzzz"},
        {"when": _SECRET},
        {"doc": _SECRET},
        {"zone": _SECRET},
    ],
)
def test_a_pydantic_message_that_carries_the_input_is_replaced(values) -> None:
    with pytest.raises(ValueError, match=r"^the config did not load:") as caught:
        _load([[Leaky]], {"leaky": values})
    message = str(caught.value)
    assert _SECRET not in message and "found `" not in message
    assert f"leaky.{next(iter(values))} (code)" in message


def test_the_servers_own_validator_text_is_kept() -> None:
    with pytest.raises(ValueError, match=r"nope keyed"):
        _load([[KeyFail]], {"keyfail": {}})


def test_a_keyed_section_that_refuses_its_defaults_names_its_key() -> None:
    with pytest.raises(ValueError, match=r"  keyfail \(default\): .*nope keyed"):
        _load([[KeyFail]], None, environ={})


def test_a_nested_default_failure_names_the_full_path() -> None:
    with pytest.raises(ValueError, match=r"  outer\.inner\.x \(default\)"):
        _load([[Outer]], None, environ={})


def test_a_root_error_with_values_given_is_from_code() -> None:
    with pytest.raises(ValueError, match=r"the root section \(code\)"):
        _load([[Failing]], {})


def test_an_int_is_named_as_an_instance() -> None:
    with pytest.raises(TypeError) as caught:
        _load([[Root]], 42)
    assert "an instance of int" in str(caught.value) and "a int" not in str(
        caught.value
    )


def test_an_instance_of_a_replaced_root_class_names_the_winner() -> None:
    with pytest.raises(TypeError, match="R2"):
        _load([[R1, R2]], [R1()])


def test_an_instance_goes_to_the_most_derived_matching_key() -> None:
    child = Child()
    assert _load([[RootOf, XOfRoot]], [child]).config.x is child


# -- schema


@config()
class Holder(Config):
    inner: Secretive = Secretive()


@config("holder")
class KeyedHolder(Config):
    inner: Secretive = Secretive()


class Serial(Config):
    token: SecretStr = Field(SecretStr("dflt"), sensitive=True)

    @model_serializer
    def _flat(self) -> str:
        return f"tok:{self.token}"


@config("serial")
class HoldsSerial(Config):
    one: Serial = Serial()
    many: list[Serial] = Field([Serial()])


@config("examples")
class Exampled(Config):
    secret: SecretStr = Field(SecretStr("x"), sensitive=True, examples=["dflt"])
    shown: str = Field("y", examples=[{"default": {"a": 1}, "k": 2}])


@config("listing")
class Listing(Config):
    inners: list[Secretive] = Field([Secretive()])
    plain: list[int] = Field([1, 2])
    pair: dict[str, int] = Field({"a": 1})


@config("named")
class Named(Config):
    default: dict[str, int] | None = None
    sensitive: str = "visible"


@pytest.mark.usefixtures("no_builtins")
def test_one_schema_covers_every_section() -> None:
    schema = config_schema(Root, Recording)
    assert schema["title"] == "Config"
    assert set(schema["properties"]) == {"name", "recording"}
    inbox = schema["$defs"]["Recording"]["properties"]["inbox"]
    assert inbox["description"] == "Where recordings go"
    assert inbox["env"] == "RECORDING_INBOX" and inbox["arg"] == "--recording-inbox"
    assert "sensitive" not in inbox


def test_every_object_forbids_unknown_keys() -> None:
    schema = config_schema(Root, Recording, Nest)
    assert schema["additionalProperties"] is False
    assert all(d["additionalProperties"] is False for d in schema["$defs"].values())


@pytest.mark.parametrize(
    "sections",
    [(Secretive,), (Holder,), (KeyedHolder,), (Listing, Secretive), (HoldsSerial,)],
)
def test_a_sensitive_default_is_never_written(sections) -> None:
    schema = config_schema(*sections)
    assert "dflt" not in json.dumps(schema)
    tokens = [
        d["properties"]["token"]
        for d in schema["$defs"].values()
        if "token" in d["properties"]
    ]
    assert tokens and all(t["sensitive"] is True for t in tokens)


def test_a_plain_default_is_kept() -> None:
    schema = config_schema(Root, Listing)
    assert schema["properties"]["name"]["default"] == "flow"
    assert schema["$defs"]["Listing"]["properties"]["plain"]["default"] == [1, 2]


def test_an_object_default_is_dropped_even_without_a_secret() -> None:
    props = config_schema(Listing)["$defs"]["Listing"]["properties"]
    assert "default" not in props["pair"]
    assert "default" not in props["inners"]


def test_a_field_named_default_or_sensitive_survives() -> None:
    props = config_schema(Named)["$defs"]["Named"]["properties"]
    assert set(props) == {"default", "sensitive"}
    assert props["sensitive"]["default"] == "visible"


def test_a_sensitive_field_loses_its_examples() -> None:
    props = config_schema(Exampled)["$defs"]["Exampled"]["properties"]
    assert "examples" not in props["secret"] and props["secret"]["sensitive"] is True


def test_data_values_are_left_intact() -> None:
    props = config_schema(Exampled)["$defs"]["Exampled"]["properties"]
    assert props["shown"]["examples"] == [{"default": {"a": 1}, "k": 2}]


def test_a_bad_section_raises_what_compose_raises() -> None:
    with pytest.raises(TypeError, match="no @config of its own"):
        config_schema(Unstamped)
    with pytest.raises(ValueError, match="claimed by both"):
        config_schema(Base, Other)


def test_the_schema_includes_the_built_ins(monkeypatch) -> None:
    monkeypatch.setattr(kubed.mcp._config, "_builtin_sections", lambda: (Base,))
    assert "base" in config_schema()["properties"]


# -- a server's config


def test_the_config_is_there_right_after_construction_and_frozen() -> None:
    mcp = QuickMCP("t", sections=[Root, Recording], config={"name": "kb"})
    assert mcp.config.name == "kb"
    with pytest.raises(ValidationError):
        mcp.config.name = "other"
    with pytest.raises(AttributeError):
        mcp.config = None  # no setter


def test_a_server_with_no_sections_has_only_the_built_ins() -> None:
    assert set(type(QuickMCP("t", environ={}).config).model_fields) == {
        "serving",
        "telemetry",
        "delivery",
    }


def test_a_bad_config_stops_construction() -> None:
    with pytest.raises(ValueError, match=r"recording\.nope \(code\)"):
        QuickMCP("t", sections=[Recording], config={"recording": {"nope": 1}})


async def test_sections_and_the_config_are_injected_by_class() -> None:
    mcp = QuickMCP("t", sections=[Root, Base, Sub], environ={})

    @mcp.tool
    def seen(
        sub: Inject[Sub], base: Inject[Base], whole: Inject[Config], root: Inject[Root]
    ) -> dict:
        return {"sub": sub is base, "whole": whole is root is mcp.config, "b": base.b}

    async with Client(mcp.server) as client:
        assert (await client.call_tool("seen", {})).data == {
            "sub": True,
            "whole": True,
            "b": 2,
        }


def test_the_composed_class_is_not_a_key() -> None:
    # The composed class isn't a key: Root and Config resolve, it doesn't.
    mcp = QuickMCP("t", sections=[Root], environ={})
    assert mcp._container.knows(Root) and mcp._container.knows(Config)
    assert not mcp._container.knows(type(mcp.config))


@config("hushed")
class Hushed(Config):
    token: SecretStr = SecretStr("hunter2")


def test_a_secret_is_kept_out_of_every_repr() -> None:
    loaded = _load([[Root, Hushed]], {"hushed": {"token": "s3cret"}}).config
    for text in (repr(loaded), str(loaded), repr(loaded.hushed), str(loaded.hushed)):
        assert "s3cret" not in text
        assert "hunter2" not in text


@config("tagged")
class Tagged(Config):
    tags: Tags


class Tags(RootModel[list[str]]):
    pass


def test_a_root_model_is_refused_with_advice_it_can_follow() -> None:
    with pytest.raises(
        TypeError, match=r"RootModel, which can't forbid unknown keys.*named fields"
    ) as c:
        _compose([[Tagged]])
    assert "extra=" not in str(c.value)


def test_sections_must_be_an_iterable() -> None:
    with pytest.raises(
        TypeError, match=r"sections= takes an iterable of section classes"
    ):
        QuickMCP("t", sections=None, environ={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"sections= takes an iterable"):
        _compose([None])  # type: ignore[list-item]


class Opaque:
    def __repr__(self) -> str:
        return "Opaque(TOPSECRET)"


@config("opaque")
class HoldsOpaque(Config):
    thing: Any = Field(Opaque())


def test_the_schema_warns_nothing_about_a_sensitive_default(recwarn) -> None:
    schema = config_schema(HoldsOpaque)
    assert "TOPSECRET" not in json.dumps(schema)
    assert not [w for w in recwarn if "TOPSECRET" in str(w.message)]


@pytest.mark.parametrize(
    "values, match",
    [
        (Settings(), "sections="),
        (argparse.Namespace(port=1), "sections="),
        (types.SimpleNamespace(delivery=None), "sections="),
        (Unregistered(), "sections="),
        ([Recording], "sections="),
    ],
)
def test_old_call_shapes_are_type_errors_through_the_server(values, match) -> None:
    with pytest.raises(TypeError, match=match):
        QuickMCP("t", sections=[Root, Recording], config=values)


def test_an_instance_in_sections_is_a_type_error_through_the_server() -> None:
    with pytest.raises(
        TypeError, match="Sections are classes, got an instance of Root"
    ):
        QuickMCP("t", sections=[Root()], environ={})
