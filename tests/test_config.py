"""The config core: sections by key, composed into one strict config model."""

from __future__ import annotations

import dataclasses
from typing import Annotated, NewType

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType, TypedDict

import kubed.mcp._config
from kubed.mcp._config import Config, Field, _compose, config

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
    monkeypatch.setattr(kubed.mcp._config, "_BUILTINS", (Base,))
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
