"""One config per server, composed from the sections the server registers.

A server's config is not one big class. It is a root section plus keyed sections, each
a pydantic model marked with ``@config``, and this module composes them into one strict
model. It imports no FastMCP, so a config can be loaded and described without a server.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import keyword
import os
import re
import types
import typing
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypeVar

import yaml
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Json,
    RootModel,
    Secret,
    SecretBytes,
    SecretStr,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic import Field as _PydanticField
from pydantic.errors import PydanticUndefinedAnnotation, PydanticUserError
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, PydanticJsonSchemaWarning

from .components import Component, _components

M = TypeVar("M", bound=BaseModel)

# The stamp ``config`` leaves on a class. A dunder name, so pydantic leaves it alone,
# and read from the class's own ``vars()``, so a subclass never inherits a key.
_STAMP = "__kubed_config_key__"

# "No default given" for ``Field``, so ``None`` stays a usable default.
_UNSET: Any = object()


class Config(BaseModel):
    """The base a section may subclass. It refuses unknown keys.

    ``Inject[Config]`` resolves to the whole loaded config.
    """

    model_config = ConfigDict(extra="forbid")


def _builtin_sections() -> tuple[type[BaseModel], ...]:
    """The sections every server has, composed before its own.

    Imported here, not at the top: ``_builtins`` defines them with this module's
    ``Config``, ``Field`` and ``config``.
    """
    from ._builtins import BUILTINS

    return BUILTINS


def _builtin_keys() -> tuple[str, ...]:
    """The built-in sections' keys, in the order they are composed."""
    return tuple(vars(cls)[_STAMP] for cls in _builtin_sections())


def _check_hooks(keys: Iterable[object], what: str) -> frozenset[str]:
    """The keys, each checked to be a built-in's key, as a set.

    ``what`` is the argument named in the error, ``"hooks="`` or ``"hooked="``.

    Raises:
        ValueError: A key isn't a built-in's.
    """
    builtins = _builtin_keys()
    checked = list(keys)
    for key in checked:
        if not isinstance(key, str) or key not in builtins:
            raise ValueError(
                f"{what} takes a built-in's key, one of "
                f"{', '.join(repr(k) for k in builtins)}; got {key!r}"
            )
    return frozenset(checked)


class _Top(Config):
    """The top of the composed config: frozen, whatever the root section says."""

    model_config = ConfigDict(frozen=True)


def _check_key(key: object) -> None:
    if not isinstance(key, str):
        raise TypeError(f"A config key must be a str, got a {type(key).__qualname__}")
    if (
        not key.isidentifier()
        or keyword.iskeyword(key)
        or key.startswith(("_", "model_"))
        or hasattr(BaseModel, key)
    ):
        raise ValueError(
            f"{key!r} can't be a config key: it must be an identifier that isn't a "
            "Python keyword, doesn't start with '_' or 'model_', and isn't an "
            "attribute of a pydantic model"
        )


def config(key: str | None = None) -> Callable[[type[M]], type[M]]:
    """Mark a pydantic model as a section of the server's config.

    ``@config()`` marks the root section, whose fields sit at the top of the config.
    ``@config("recording")`` marks a keyed section, which sits under that key. The class
    is returned unchanged, and the server picks it up through ``sections=``.

    A subclass doesn't inherit its parent's mark: a subclass that should replace its
    parent carries its own ``@config(key)``. A section may be any ``BaseModel`` that
    forbids unknown keys, and ``Config`` is one.

    The composed top level is frozen, and so a ``mode="after"`` validator on the root
    section can't assign to ``self``. Keyed sections aren't frozen, so their validators
    can.

    Args:
        key: The section's key, an identifier that isn't a Python keyword, doesn't
            start with ``_`` or ``model_`` and isn't an attribute of ``BaseModel``.
            ``None`` is the root.

    Raises:
        TypeError: The key isn't a ``str``, the decorator was written without
            parentheses, or the decorated class isn't a ``BaseModel`` subclass.
        ValueError: The key can't name a section.
    """
    if isinstance(key, type):
        raise TypeError(
            f"Write @config(), with parentheses, to mark {key.__qualname__} "
            "as the root section"
        )
    if key is not None:
        _check_key(key)

    def stamp(cls: type[M]) -> type[M]:
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            kind = cls.__qualname__ if isinstance(cls, type) else type(cls).__qualname__
            raise TypeError(
                f"@config marks a pydantic BaseModel subclass, got {kind}"
                f"{'' if isinstance(cls, type) else ' (an instance)'}"
            )
        setattr(cls, _STAMP, key)
        return cls

    return stamp


def Field(
    default: Any = _UNSET,
    *,
    doc: str | None = None,
    env: str | None = None,
    arg: str | None = None,
    sensitive: bool = False,
    **kwargs: Any,
) -> Any:
    """Pydantic's ``Field``, with the config item's metadata.

    ``doc`` is the field's description. ``env`` is the environment variable and ``arg``
    the command-line flag that set it, taken literally: nothing derives or prefixes
    them. A sensitive item is one whose type is a secret type, ``SecretStr``,
    ``SecretBytes`` or ``Secret[T]``: pydantic masks it in ``repr``, ``str`` and every
    dump, ``get_secret_value()`` reads it, and the schema never writes its default.
    ``sensitive=True`` says so, and is refused at composition on a field with no
    secret type. They are written to the field's ``json_schema_extra``, and only the
    ones given. Every other keyword goes to pydantic.

    Raises:
        TypeError: ``env`` or ``arg`` isn't a non-empty ``str``, or ``description``
            or ``json_schema_extra`` is passed, which ``doc=`` and the item keywords
            replace.
    """
    for name in ("description", "json_schema_extra"):
        if name in kwargs:
            raise TypeError(
                f"Field() takes doc=, env=, arg= and sensitive= in place of {name}="
            )
    extra: dict[str, Any] = {}
    for name, value in (("env", env), ("arg", arg)):
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise TypeError(
                f"Field({name}=) must be a non-empty str, got a "
                f"{type(value).__qualname__}"
            )
        extra[name] = value
    if sensitive:
        extra["sensitive"] = True
    if doc is not None:
        kwargs["description"] = doc
    if extra:
        kwargs["json_schema_extra"] = extra
    if default is _UNSET:
        return _PydanticField(**kwargs)
    return _PydanticField(default, **kwargs)


class NoDecode:
    """Marks a field whose env var or flag reaches its validators as the raw text.

    Written as ``Annotated[list[str], NoDecode]``. Without it, a list, tuple, set, dict
    or model field reads its env var or flag as JSON. With it, the text is passed as it
    is, so the field's own ``mode="before"`` validator can split it as it likes.
    """


class _Item(NamedTuple):
    """One field of the composed config, reached from the top through model fields.

    ``loc`` is the keys its value is validated under, ``attrs`` the attribute names
    that read it back from the loaded config, and ``path`` the dotted ``loc``. A
    ``leaf`` is a field the walk doesn't descend into. ``env`` and ``arg`` are the names
    that set it, as written, ``sensitive`` says a secret type is in its annotation or,
    on a leaf, a model with a secret field at any depth, and ``takes`` says how a raw
    value from an env var or a flag decodes.
    """

    loc: tuple[str, ...]
    attrs: tuple[str, ...]
    path: str
    leaf: bool
    env: str | None
    arg: str | None
    sensitive: bool
    takes: Literal["string", "array", "object"]


class _Composition(NamedTuple):
    """What ``_compose`` returns: the model, who holds and lost each key, its items."""

    model: type[Config]
    held: dict[str | None, type[BaseModel]]
    replaced: dict[str | None, tuple[type[BaseModel], ...]]
    items: tuple[_Item, ...]


def _where(key: str | None) -> str:
    return "the root section" if key is None else f"the section {key!r}"


def _register(
    cls: Any,
    held: dict[str | None, type[BaseModel]],
    replaced: dict[str | None, list[type[BaseModel]]],
) -> None:
    if not isinstance(cls, type):
        raise TypeError(
            f"Sections are classes, got an instance of {type(cls).__qualname__}: "
            "values for the config go in config="
        )
    if _STAMP not in vars(cls):
        raise TypeError(
            f"{cls.__qualname__} has no @config of its own: add @config(...) to it"
        )
    key = vars(cls)[_STAMP]
    holder = held.get(key)
    if holder is None:
        held[key] = cls
    elif cls is holder or cls in replaced.get(key, ()):
        return
    elif issubclass(cls, holder):
        replaced.setdefault(key, []).append(holder)
        held[key] = cls
    elif issubclass(holder, cls):
        replaced.setdefault(key, []).append(cls)
    else:
        raise ValueError(
            f"{_where(key)} is claimed by both {holder.__qualname__} and "
            f"{cls.__qualname__}; one must subclass the other"
        )


def _parts(annotation: object) -> Iterable[object]:
    """Every node of an annotation: itself, then the arguments of its generics.

    A NewType and a type alias are followed to what they name, and a subscripted alias
    to its arguments too. ``Annotated`` metadata, such as a nested ``FieldInfo``, is a
    node too.
    """
    if hasattr(annotation, "__supertype__"):  # a NewType
        yield from _parts(annotation.__supertype__)
        return
    if hasattr(annotation, "__value__") and not isinstance(annotation, type):
        # A TypeAliasType, or one subscripted, as Many[Login]: its body holds a
        # TypeVar where the subscript holds the type, so both are followed.
        yield from _parts(annotation.__value__)
        for arg in typing.get_args(annotation):
            yield from _parts(arg)
        return
    yield annotation
    for arg in typing.get_args(annotation):
        yield from _parts(arg)


def _models_in(annotation: object) -> Iterable[type]:
    """The classes an annotation reaches: itself, and the arguments of its generics."""
    return (part for part in _parts(annotation) if isinstance(part, type))


def _is_secret(annotation: object) -> bool:
    """Whether an annotation is a secret type, which pydantic masks wherever it prints.

    That is ``SecretStr``, ``SecretBytes``, ``Secret[T]``, or a subclass of any of them.
    """
    origin = typing.get_origin(annotation) or annotation
    return isinstance(origin, type) and issubclass(
        origin, (SecretStr, SecretBytes, Secret)
    )


# Per class, as pydantic's own per-class data is. Weak, so a composed model that's
# dropped isn't kept alive.
_SENSITIVE: weakref.WeakKeyDictionary[type, frozenset[str]] = (
    weakref.WeakKeyDictionary()
)


def _sensitive_fields(cls: type[BaseModel]) -> frozenset[str]:
    """The names of ``cls``'s fields with a secret type anywhere in the annotation.

    ``SecretStr | None``, ``list[SecretStr]`` and ``dict[str, SecretStr]`` all count. A
    field that holds another model isn't sensitive for that model's fields.
    """
    names = _SENSITIVE.get(cls)
    if names is None:
        names = frozenset(
            name
            for name, info in cls.model_fields.items()
            if any(_is_secret(part) for part in _parts(info.annotation))
        )
        _SENSITIVE[cls] = names
    return names


def _holds_secret(annotation: object) -> bool:
    """Whether an annotation reaches a model with a secret field, at any depth.

    Such a model is dumped by its own serializer, which may write its secret as
    anything, so a leaf holding it is shown as a secret.
    """
    queue = [m for m in _models_in(annotation) if issubclass(m, BaseModel)]
    seen = set(queue)
    while queue:
        cls = queue.pop()
        if _sensitive_fields(cls):
            return True
        for info in cls.model_fields.values():
            for model in _models_in(info.annotation):
                if issubclass(model, BaseModel) and model not in seen:
                    seen.add(model)
                    queue.append(model)
    return False


# The keywords ``Field`` writes that only mean something on the field itself.
_ITEM_KEYWORDS = ("env", "arg", "sensitive")


def _check_item_keywords(cls: type[BaseModel], name: str, path: str) -> None:
    """Refuse item keywords that would silently do nothing, or a false ``sensitive``.

    A name on a field whose alias isn't a ``str`` is refused too: the name's value is
    written under one key, and such a field takes its value under others.
    """
    info = cls.model_fields[name]
    alias = info.validation_alias
    named = [
        k
        for k in ("env", "arg")
        if isinstance(info.json_schema_extra, dict) and k in info.json_schema_extra
    ]
    if named and not (alias is None or isinstance(alias, str)):
        written = " and ".join(f"{k}=" for k in named)
        raise TypeError(
            f"{path}: a named item takes a plain string alias or none, not an "
            f"{type(alias).__name__}: drop its {written} or its validation_alias"
        )
    for part in _parts(info.annotation):
        if isinstance(part, FieldInfo) and isinstance(part.json_schema_extra, dict):
            given = [k for k in _ITEM_KEYWORDS if k in part.json_schema_extra]
            if given:
                written = " and ".join(f"{k}=" for k in given)
                raise TypeError(
                    f"{path}: {written} in a Field inside the annotation is never "
                    "read: put Field(...) on the field itself"
                )
    extra = info.json_schema_extra
    if (
        isinstance(extra, dict)
        and extra.get("sensitive")
        and name not in _sensitive_fields(cls)
    ):
        raise TypeError(
            f"{path}: sensitive=True needs a secret type: use SecretStr, "
            "SecretBytes or Secret[T]"
        )
    _check_inherited_names(cls, name, path)


def _names_of(info: FieldInfo) -> dict[str, str]:
    """The ``env`` and ``arg`` a field was given, by keyword."""
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    return {k: extra[k] for k in ("env", "arg") if k in extra}


def _check_inherited_names(cls: type[BaseModel], name: str, path: str) -> None:
    """Refuse a field that drops an ``env`` or ``arg`` a model base's field gave it.

    Pydantic drops a parent field's metadata when a subclass redefines the field, so
    a bare redefinition would lose the names that set it, at any depth of the chain.
    Each keyword is compared on its own, with the nearest model base whose field of
    the same name has that keyword. The error lists every keyword dropped and names
    the nearest base that has any of them. Restating either under another name is
    allowed.
    """
    kept = _names_of(cls.model_fields[name])
    dropped: dict[str, str] = {}
    nearest: type[BaseModel] | None = None
    for base in cls.__mro__[1:]:
        if not (isinstance(base, type) and issubclass(base, BaseModel)):
            continue
        field = base.model_fields.get(name)
        given = {} if field is None else _names_of(field)
        for kind in ("env", "arg"):
            if kind in kept or kind in dropped or kind not in given:
                continue
            dropped[kind] = given[kind]
            nearest = nearest or base
    if nearest is None:
        return
    # In the keyword order, whichever base each one came from.
    dropped = {k: dropped[k] for k in ("env", "arg") if k in dropped}
    written = " and ".join(f"{k}={v!r}" for k, v in dropped.items())
    restated = ", ".join(f"{k}={v!r}" for k, v in dropped.items())
    them = "them" if len(dropped) > 1 else "it"
    raise TypeError(
        f"{path}: redefines {nearest.__qualname__}.{name} without its "
        f"{written}: restate Field(..., {restated}) to keep {them}"
    )


def _is_typed_dict(cls: object) -> bool:
    """Whether ``cls`` is a TypedDict, from ``typing`` or from ``typing_extensions``.

    Before Python 3.12 ``typing.is_typeddict`` doesn't know the latter's classes, so
    the check is structural: a ``dict`` subclass that carries the TypedDict key sets.
    """
    return typing.is_typeddict(cls) or (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__required_keys__")
        and hasattr(cls, "__optional_keys__")
    )


def _check_strict(
    cls: type,
    path: str,
    seen: set[type],
) -> None:
    if cls in seen:
        return
    seen.add(cls)
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        if issubclass(cls, RootModel):
            raise TypeError(
                f"{cls.__qualname__} at {path} is a RootModel, which can't forbid "
                "unknown keys: use a model with named fields"
            )
        if cls.model_config.get("extra") != "forbid":
            raise TypeError(
                f"{cls.__qualname__} at {path} allows unknown keys: "
                "give it model_config = ConfigDict(extra='forbid')"
            )
        if not cls.__pydantic_complete__:
            try:
                cls.model_rebuild(raise_errors=True)
            except (PydanticUndefinedAnnotation, PydanticUserError) as exc:
                raise TypeError(
                    f"{cls.__qualname__} at {path} has an annotation that doesn't "
                    "resolve, so its fields can't be checked"
                ) from exc
        prefix = "" if path == "the root section" else f"{path}."
        for name, info in cls.model_fields.items():
            _check_item_keywords(cls, name, f"{prefix}{name}")
            for inner in _models_in(info.annotation):
                _check_strict(inner, f"{prefix}{name}", seen)
    elif dataclasses.is_dataclass(cls) or _is_typed_dict(cls):
        raise TypeError(
            f"{cls.__qualname__} at {path} is a dataclass or TypedDict, which doesn't "
            "refuse unknown keys: make it a pydantic model that forbids them"
        )


def _strip(annotation: object) -> object:
    """The annotation under any ``Annotated``, NewType and type alias wrapping it."""
    while True:
        if typing.get_origin(annotation) is Annotated:
            annotation = typing.get_args(annotation)[0]
        elif hasattr(annotation, "__supertype__"):
            annotation = annotation.__supertype__
        elif hasattr(annotation, "__value__") and not isinstance(annotation, type):
            annotation = annotation.__value__
        else:
            return annotation


def _members(annotation: object) -> tuple[object, ...] | None:
    """A union's members other than ``None``, or ``None`` when it isn't a union."""
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return None
    return tuple(m for m in typing.get_args(annotation) if m is not type(None))


def _model_typed(annotation: object) -> type[BaseModel] | None:
    """The one model class a field holds, past ``Annotated`` and a ``None``, or None.

    A union of several models, a list, dict or tuple of models and a secret aren't
    model-typed.
    """
    annotation = _strip(annotation)
    members = _members(annotation)
    if members is not None:
        if len(members) != 1:
            return None
        annotation = _strip(members[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _kind(annotation: object) -> Literal["string", "array", "object"]:
    """How a raw value for ``annotation`` decodes: as text, a JSON array or object."""
    annotation = _strip(annotation)
    members = _members(annotation)
    if members is not None:
        kinds = {_kind(m) for m in members}
        return kinds.pop() if len(kinds) == 1 else "string"
    origin = typing.get_origin(annotation) or annotation
    if not isinstance(origin, type):
        return "string"
    if issubclass(origin, Secret):  # Secret[T] decodes as T
        args = typing.get_args(annotation) or next(
            (
                typing.get_args(base)
                for base in getattr(origin, "__orig_bases__", ())
                if isinstance(typing.get_origin(base), type)
                and issubclass(typing.get_origin(base), Secret)
            ),
            (),
        )
        return _kind(args[0]) if args else "string"
    if issubclass(origin, (SecretStr, SecretBytes, str, bytes, bytearray)):
        return "string"
    if issubclass(origin, (BaseModel, Mapping)):
        return "object"
    if issubclass(origin, (Sequence, AbstractSet)):
        return "array"
    return "string"


def _takes(info: FieldInfo) -> Literal["string", "array", "object"]:
    """How a field's raw value decodes. ``NoDecode`` anywhere on it makes it text.

    So does pydantic's ``Json``, whose validator parses the raw string itself and
    refuses one already decoded.
    """
    marks = (*info.metadata, *_parts(info.annotation))
    if any(m in (NoDecode, Json) or isinstance(m, (NoDecode, Json)) for m in marks):
        return "string"
    return _kind(info.annotation)


def _walk_items(
    cls: type[BaseModel],
    loc: tuple[str, ...],
    attrs: tuple[str, ...],
    on_path: frozenset[type],
    items: list[_Item],
    direct: set[type],
    held_in: dict[type, str],
) -> None:
    """Add an item for each of ``cls``'s fields, descending into model-typed ones.

    ``direct`` collects the models descended into. ``held_in`` collects, for each model
    a leaf's annotation reaches, the path of the first leaf that holds it.
    """
    for name, info in cls.model_fields.items():
        here = (*loc, _key(cls, name, info))
        names = (*attrs, name)
        path = ".".join(here)
        inner = _model_typed(info.annotation)
        leaf = inner is None or inner in on_path
        extra = info.json_schema_extra
        if not isinstance(extra, dict):
            extra = {}
        items.append(
            _Item(
                loc=here,
                attrs=names,
                path=path,
                leaf=leaf,
                env=extra.get("env"),
                arg=extra.get("arg"),
                sensitive=name in _sensitive_fields(cls)
                or (leaf and _holds_secret(info.annotation)),
                takes=_takes(info),
            )
        )
        if inner is not None and not leaf:
            direct.add(inner)
            _walk_items(inner, here, names, on_path | {inner}, items, direct, held_in)
        else:
            for model in _models_in(info.annotation):
                if issubclass(model, BaseModel):
                    held_in.setdefault(model, path)


# A flag is ``--`` and a name that doesn't start with ``-``, with no space or ``=``.
_FLAG = re.compile(r"--[^-\s=][^\s=]*")
# The names quick-mcp reads itself, for the config file.
_RESERVED = {"env": "CONFIG_FILE", "arg": "--config-file"}
_NAME_KIND = {"env": "env var", "arg": "flag"}


def _is_hooked(item: _Item, hooked: AbstractSet[str]) -> bool:
    """Whether an item sits under a hooked key, so no name of it is read."""
    return item.loc[0] in hooked


def _check_names(
    items: Iterable[_Item], hooked: AbstractSet[str] = frozenset()
) -> None:
    """Refuse a malformed name, one of quick-mcp's own, or one declared twice.

    The items under a hooked key are skipped: their names are never read. A name
    declared twice where one side is under a built-in's key says to hook that
    built-in, which takes its names out of the check.
    """
    builtins = _builtin_keys()
    declared: dict[tuple[str, str], _Item] = {}
    for item in items:
        if _is_hooked(item, hooked):
            continue
        for kind, name in (("env", item.env), ("arg", item.arg)):
            if name is None:
                continue
            if kind == "arg" and not (isinstance(name, str) and _FLAG.fullmatch(name)):
                raise TypeError(
                    f"{item.path}: arg={name!r} isn't a flag: a flag starts with '--', "
                    "then a name with no space or '='"
                )
            if kind == "env" and not (
                isinstance(name, str) and name and "=" not in name and "\0" not in name
            ):
                raise TypeError(
                    f"{item.path}: env={name!r} isn't an env var name: it must be a "
                    "non-empty str with no '=' or NUL character"
                )
            if name == _RESERVED[kind]:
                raise ValueError(
                    f"{item.path}: the {_NAME_KIND[kind]} {name} is quick-mcp's own: "
                    "it reads it for the config file"
                )
            first = declared.setdefault((kind, name), item)
            if first.path != item.path:
                built_in = next(
                    (i.loc[0] for i in (first, item) if i.loc[0] in builtins), None
                )
                hint = (
                    ""
                    if built_in is None
                    else "; to keep your own, hook the built-in: "
                    f"hooks={{{built_in!r}: ...}}"
                )
                raise ValueError(
                    f"The {_NAME_KIND[kind]} {name!r} is declared by both {first.path} "
                    f"and {item.path}{hint}"
                )


def _check_held(held_in: dict[type, str], direct: set[type]) -> None:
    """Refuse a name on a field of a model reached only through a container or union.

    Such a field has no one path a value could be set at. A model also reached as a
    model-typed field keeps its names, which set it there. The models below a held
    model, at any depth, are held too.
    """
    queue = list(held_in)
    while queue:
        cls = queue.pop(0)
        for info in cls.model_fields.values():
            for model in _models_in(info.annotation):
                if issubclass(model, BaseModel) and model not in held_in:
                    held_in[model] = held_in[cls]
                    queue.append(model)
    for cls, path in held_in.items():
        if cls in direct:
            continue
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            if not isinstance(extra, dict):
                continue
            for kind in ("env", "arg"):
                if kind in extra:
                    raise TypeError(
                        f"{path}: {cls.__qualname__}.{name} can't be set by name, as "
                        f"{cls.__qualname__} is reached only through a list, dict, "
                        f"tuple or union: drop its {kind}={extra[kind]!r}"
                    )


def _items(
    model: type[BaseModel],
    root: type[BaseModel] | None,
    hooked: AbstractSet[str] = frozenset(),
) -> tuple[_Item, ...]:
    """The composed model's items, depth first in declaration order, checked.

    The names of the items under a hooked key aren't checked, as they aren't read.
    """
    items: list[_Item] = []
    direct: set[type] = {model} if root is None else {model, root}
    held_in: dict[type, str] = {}
    _walk_items(model, (), (), frozenset({model}), items, direct, held_in)
    _check_names(items, hooked)
    _check_held(held_in, direct)
    return tuple(items)


_NOT_ITERABLE = "sections= takes an iterable of section classes, such as a list"


def _compose(
    sources: Iterable[Iterable[type[BaseModel]]],
    *,
    hooked: frozenset[str] = frozenset(),
) -> _Composition:
    """Register the sections of every source and compose them into one config model.

    The built-in sections come first, then each source in order. The most derived class
    wins a key. Every field reached through model fields is recorded as an item, and the
    names items declare are checked here, so ``config_schema`` refuses what a server
    refuses. The names under a ``hooked`` key aren't checked, as a hook supplies that
    section and they are never read.
    """
    held: dict[str | None, type[BaseModel]] = {}
    replaced: dict[str | None, list[type[BaseModel]]] = {}
    if not isinstance(sources, Iterable):
        raise TypeError(_NOT_ITERABLE)
    for source in (_builtin_sections(), *sources):
        if isinstance(source, BaseModel):
            raise TypeError(
                f"sections= takes classes, got an instance of "
                f"{type(source).__qualname__}: values for the config go in config="
            )
        if isinstance(source, type):
            raise TypeError(
                "sections= takes classes in a sequence: write "
                f"sections=({source.__qualname__},)"
            )
        if not isinstance(source, Iterable):
            raise TypeError(_NOT_ITERABLE)
        for cls in source:
            _register(cls, held, replaced)

    seen: set[type] = set()
    for key, cls in held.items():
        _check_strict(cls, "the root section" if key is None else key, seen)

    root = held.get(None)
    fields: dict[str, Any] = {}
    for key, cls in held.items():
        if key is None:
            continue
        if root is not None and key in root.model_fields:
            raise ValueError(
                f"The key {key!r} is a field of the root section "
                f"{root.__qualname__}, and the key of {cls.__qualname__}"
            )
        if root is not None and hasattr(root, key):
            raise ValueError(
                f"The key {key!r} is an attribute of the root section "
                f"{root.__qualname__}, and the key of {cls.__qualname__}"
            )
        if all(not f.is_required() for f in cls.model_fields.values()):
            # Validate ``{}`` rather than call ``cls()``, so a failing default is
            # reported under the key.
            fields[key] = (
                cls,
                _PydanticField(default_factory=dict, validate_default=True),
            )
        else:
            fields[key] = (cls, ...)

    bases = (root, _Top) if root is not None else (_Top,)
    model = create_model("Config", __base__=bases, __module__=__name__, **fields)
    return _Composition(
        model,
        held,
        {key: tuple(losers) for key, losers in replaced.items()},
        _items(model, root, hooked),
    )


class Source(NamedTuple):
    """Where a config item's value came from.

    ``kind`` is an open set: ``"default"``, ``"file"``, ``"env"``, ``"arg"``,
    ``"code"`` and ``"hook"`` today. ``name`` is the file's path, the env var or the
    flag, else ``None``. As a string it is the kind, then the name when there is one:
    ``default``, ``file /etc/kb.yaml``, ``env PORT``, ``arg --port``.
    """

    kind: str
    name: str | None = None

    def __str__(self) -> str:
        return self.kind if self.name is None else f"{self.kind} {self.name}"


class _Loaded(NamedTuple):
    """What ``_load`` returns: the config, what to inject for each class, its sources
    and its boot line.

    ``sources`` maps each leaf item's path to where its value came from. ``boot_line``
    names each item not at its default, its value and its source: the server logs it
    each time serving starts.
    """

    config: Config
    provided: dict[type, object]
    sources: Mapping[str, Source]
    boot_line: str


def _bad_values(what: str) -> TypeError:
    return TypeError(
        f"config= takes a mapping or section instances, got {what}; "
        "classes go in sections="
    )


def _spread(
    items: Iterable[Any],
    held: dict[str | None, type[BaseModel]],
    replaced: dict[str | None, tuple[type[BaseModel], ...]],
) -> dict[str, Any]:
    """Turn section instances into one mapping, each under the key it belongs to."""
    data: dict[str, Any] = {}
    seen: set[str | None] = set()
    for item in items:
        if isinstance(item, type):
            raise _bad_values(f"the class {item.__qualname__}")
        if not isinstance(item, BaseModel):
            raise _bad_values(f"an instance of {type(item).__qualname__}")
        kind = type(item)
        exact = [key for key, cls in held.items() if cls is kind]
        matches = exact or [key for key, cls in held.items() if isinstance(item, cls)]
        # The most derived winning class takes an instance several of them match.
        matches.sort(key=lambda key: len(held[key].__mro__), reverse=True)
        if not matches:
            for key, losers in replaced.items():
                if isinstance(item, losers):
                    raise TypeError(
                        f"{kind.__qualname__} lost {_where(key)} to "
                        f"{held[key].__qualname__}: give an instance of "
                        f"{held[key].__qualname__}"
                    )
            raise _bad_values(
                f"an instance of {kind.__qualname__}, which isn't registered"
            )
        key = matches[0]
        if key in seen:
            raise ValueError(f"Two instances were given for {_where(key)}")
        seen.add(key)
        if key is None:
            data.update({name: getattr(item, name) for name in kind.model_fields})
        else:
            data[key] = item
    return data


def _path(loc: tuple[int | str, ...], model: type[BaseModel]) -> str:
    """Join a pydantic ``loc``: names with ``.``, indexes as ``[i]``, up to a secret.

    The walk keeps a set of model classes that only grows, starting with ``model``. A
    ``str`` part that is a first key of a field of any class in the set (``_accepts``)
    is written. If that field is sensitive the path ends after it, so no dict key or
    index below a secret prints, at any depth. Otherwise the model classes the field's
    annotation reaches,
    through ``Optional``, ``Annotated``, unions, lists, tuples, sets and dict values,
    join the set. Any other part, an index, a dict key or a union tag, is written and
    leaves the set as it is. A dict key that is also a field's name is taken for the
    field, and the set never shrinks, so a path may stop early but never late.
    """
    if not loc:
        return "the root section"
    path = ""
    at: tuple[type[BaseModel], ...] = (model,)
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
            continue
        if part.startswith("["):  # pydantic's "[key]", for a bad dict key
            path += part
        else:
            path += f".{part}" if path else part
        fields = [
            (cls, name)
            for cls in at
            for name, info in cls.model_fields.items()
            if part in _accepts(cls, name, info)
        ]
        if not fields:
            continue
        if any(name in _sensitive_fields(cls) for cls, name in fields):
            break
        reached = (
            inner
            for cls, name in fields
            for inner in _models_in(cls.model_fields[name].annotation)
            if issubclass(inner, BaseModel)
        )
        at = tuple(dict.fromkeys((*at, *reached)))
    return path


class _Layer(NamedTuple):
    """One source's values, keyed as the config, and the names that set them.

    ``names`` holds a ``(loc, name)`` pair for each contribution: the file layer has
    one, at ``()``, named by its path, values in code one at ``()`` with no name, and
    the hooks one per hooked key, at ``(key,)`` with no name.
    """

    kind: str
    data: dict[str, Any]
    names: tuple[tuple[tuple[str, ...], str | None], ...]


def _holds(data: Any, loc: tuple[int | str, ...]) -> bool:
    """Whether a layer's data reaches ``loc``, or a value that isn't a mapping above it.

    A list, a scalar or a model instance holds everything below it.
    """
    node = data
    for part in loc:
        if not isinstance(node, Mapping):
            return True
        if part not in node:
            return False
        node = node[part]
    return True


def _source(loc: tuple[int | str, ...], layers: Sequence[_Layer]) -> Source:
    """Where the value at ``loc`` came from: the highest layer holding it.

    Its name is the deepest contribution whose loc is a prefix of ``loc``, else the one
    contribution below ``loc`` when there is exactly one, else ``None``. No layer
    holding ``loc`` is the default.
    """
    for layer in reversed(layers):
        if not _holds(layer.data, loc):
            continue
        above = [(at, name) for at, name in layer.names if loc[: len(at)] == at]
        if above:
            return Source(layer.kind, max(above, key=lambda c: len(c[0]))[1])
        below = [name for at, name in layer.names if at[: len(loc)] == loc]
        return Source(layer.kind, below[0] if len(below) == 1 else None)
    return Source("default")


def _blank(value: object) -> bool:
    """Whether a value means unset: ``None``, or a string of only whitespace."""
    return value is None or (isinstance(value, str) and not value.strip())


def _unblank(data: Mapping[str, Any]) -> dict[str, Any]:
    """A file's mapping without its blank entries, at every depth of nested mappings.

    A list and what it holds are left to the validators.
    """
    return {
        key: _unblank(value) if isinstance(value, Mapping) else value
        for key, value in data.items()
        if not _blank(value)
    }


def _merge(low: Any, high: Any) -> Any:
    """``high`` over ``low``: mappings key by key at every depth, else ``high`` whole.

    A model instance under a mapping counts as its fields, keyed as it validates, so
    the keys the mapping doesn't set keep the instance's values. A field the mapping
    sets under another key it accepts takes the mapping's value alone, and one that
    accepts no key a value can be written under takes its class default.
    """
    if not isinstance(high, Mapping):
        return high
    if isinstance(low, BaseModel):
        instance, cls = low, type(low)
        low = {}
        for name, info in cls.model_fields.items():
            key, accepted = _key(cls, name, info), _accepts(cls, name, info)
            if key in accepted and not any(k in high for k in accepted if k != key):
                low[key] = getattr(instance, name)
    if not isinstance(low, Mapping):
        return high
    merged = dict(low)
    for key, value in high.items():
        merged[key] = _merge(merged[key], value) if key in merged else value
    return merged


def _by_alias(cls: type[BaseModel]) -> bool:
    return cls.model_config.get("validate_by_alias", True) is not False


def _alias_key(name: str, info: FieldInfo) -> str:
    """The one top-level key a field's value is written under, to be read by alias.

    That is its ``str`` alias, else the first of its alias choices that is a ``str``
    or an ``AliasPath`` of one ``str``, else its name.
    """
    alias = info.validation_alias
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
    for choice in choices:
        if isinstance(choice, str):
            return choice
        if (
            isinstance(choice, AliasPath)
            and len(choice.path) == 1
            and isinstance(choice.path[0], str)
        ):
            return choice.path[0]
    return name


def _schema_key(name: str, info: FieldInfo) -> str:
    """The property a field is under in pydantic's schema.

    Pydantic's own rule: its ``str`` alias, else the first of its ``AliasChoices`` that
    is a ``str`` or an ``AliasPath`` of one ``str``, else its name. A bare
    ``AliasPath`` gives the name, even one of a single step.
    """
    alias = info.validation_alias
    if isinstance(alias, AliasChoices):
        return _alias_key(name, info)
    return alias if isinstance(alias, str) else name


def _key(cls: type[BaseModel], name: str, info: FieldInfo) -> str:
    """The one key a field of ``cls`` is written under, which it's read back from.

    A model that doesn't validate by alias takes the field's name.
    """
    return _alias_key(name, info) if _by_alias(cls) else name


def _accepts(cls: type[BaseModel], name: str, info: FieldInfo) -> tuple[str, ...]:
    """Every first key a field of ``cls`` takes its value under.

    That is its ``str`` alias, the first element of an ``AliasPath`` and of each
    ``AliasChoices`` choice, and its name when it has no alias or ``cls`` validates by
    name (``validate_by_name`` or ``populate_by_name``). A model that doesn't validate
    by alias takes the name alone.
    """
    alias = info.validation_alias
    if alias is None or not _by_alias(cls):
        return (name,)
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
    keys = [c if isinstance(c, str) else c.path[0] for c in choices]
    settings = cls.model_config
    if settings.get("validate_by_name") or settings.get("populate_by_name"):
        keys.append(name)
    return tuple(dict.fromkeys(k for k in keys if isinstance(k, str)))


def _default_instance(info: FieldInfo) -> BaseModel | None:
    """A field's default when it's a model instance, or a no-argument factory makes one.

    A ``default_factory`` that takes no argument is called, once. One that takes the
    validated data is never called.
    """
    if isinstance(info.default, BaseModel):
        return info.default
    if info.default_factory is None or info.default_factory_takes_validated_data:
        return None
    made = info.get_default(call_default_factory=True)
    return made if isinstance(made, BaseModel) else None


def _over_defaults(model: type[BaseModel], data: Mapping[str, Any]) -> dict[str, Any]:
    """``data`` over the defaults that are model instances, at every depth.

    A model-typed field whose data is a mapping, under any key the field accepts, and
    whose default is an instance, gets the instance's values for the keys the data
    leaves out. A ``default_factory`` that takes no argument is called once, only then,
    and its result merged over the same way when it is an instance. Pydantic calls it
    itself when the field is absent, so it runs once per load either way. A factory that
    takes the validated data is never called here.
    """
    merged = dict(data)
    for name, info in model.model_fields.items():
        inner = _model_typed(info.annotation)
        if inner is None:
            continue
        key = next((k for k in _accepts(model, name, info) if k in merged), None)
        if key is None or not isinstance(merged[key], Mapping):
            continue
        value = merged[key]
        default = _default_instance(info)
        if default is not None:
            value = _merge(default, value)
        merged[key] = _over_defaults(inner, value)
    return merged


def _nest(loc: tuple[str, ...], value: Any) -> dict[str, Any]:
    """``value`` under the keys of ``loc``, as one nested mapping."""
    for part in reversed(loc[1:]):
        value = {part: value}
    return {loc[0]: value}


_JSON_KIND = {"array": list, "object": dict}


def _decode(raw: Any, takes: Literal["string", "array", "object"]) -> tuple[Any, bool]:
    """A raw env var or flag value as its item takes it, and whether it decoded.

    Text is passed as it is, to pydantic's lax validation. A list, set, tuple, dict or
    model reads JSON of its kind.
    """
    if takes == "string":
        return raw, True
    try:
        value = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None, False
    return value, isinstance(value, _JSON_KIND[takes])


def _named_layer(
    kind: str, raws: Iterable[tuple[int, _Item, str, Any]]
) -> tuple[_Layer, list[str]]:
    """The layer of the env vars or the flags that were set, and their decode errors.

    ``raws`` is ``(index, item, name, raw)`` in item order. A blank value is dropped
    first. The contributions merge by the depth of their item, then in item order, so
    a name on a nested field beats one on the model field holding it.
    """
    decoded: list[tuple[int, _Item, str, Any]] = []
    errors: list[str] = []
    for index, item, name, raw in raws:
        if _blank(raw):
            continue
        value, ok = _decode(raw, item.takes)
        if not ok:
            errors.append(
                f"  {item.path} ({Source(kind, name)}): must be a JSON {item.takes}"
            )
            continue
        decoded.append((index, item, name, value))
    decoded.sort(key=lambda each: (len(each[1].loc), each[0]))
    data: dict[str, Any] = {}
    for _, item, _, value in decoded:
        data = _merge(data, _nest(item.loc, value))
    names = tuple((item.loc, name) for _, item, name, _ in decoded)
    return _Layer(kind, data, names), errors


def _read_env(
    items: Sequence[_Item], environ: Mapping[str, str]
) -> tuple[_Layer, list[str]]:
    """The env layer: each item's declared env var, read by its exact name."""
    return _named_layer(
        "env",
        (
            (index, item, item.env, environ.get(item.env))
            for index, item in enumerate(items)
            if item.env is not None
        ),
    )


class _ArgvError(Exception):
    """What argparse would have printed before exiting."""


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises where argparse would exit."""

    def error(self, message: str) -> typing.NoReturn:
        raise _ArgvError(message)


_POSITIONAL = (
    "  the arguments (arg): no item takes a positional argument, got {}; "
    "a server passes argv=sys.argv[1:]"
)


def _flag_in(token: str) -> str | None:
    """A token's text up to any ``=`` when that has a flag's shape, else ``None``.

    A flag holds no space either, so a token argparse kept whole is cut there too.
    """
    flag = re.split(r"[=\s]", token, maxsplit=1)[0]
    return flag if _FLAG.fullmatch(flag) else None


def _read_argv(
    items: Sequence[_Item], argv: Sequence[str]
) -> tuple[dict[int, str], str | None, list[str], bool]:
    """Parse ``argv`` into each item's raw flag value, by item index.

    Returns the values, the ``--config-file`` value, the argv errors and whether
    argparse itself failed. A leftover with a flag's shape up to any ``=`` is an
    unknown flag, reported by that text. Without an ``=``, the token after it is its
    value when that has no flag's shape, even one starting with ``-``, and is never
    shown. Every other leftover, a short flag or a bare ``-`` included, is counted
    with the positionals and never quoted. So is every token after a bare ``--``,
    which argparse never sees, whatever its version does with one.
    """
    argv = list(argv)
    end = argv.index("--") if "--" in argv else len(argv)
    argv, after = argv[:end], argv[end + 1 :]
    parser = _Parser(
        prog="server",  # never read from sys.argv
        add_help=False,
        allow_abbrev=False,
        argument_default=argparse.SUPPRESS,
    )
    for index, item in enumerate(items):
        if item.arg is not None:
            parser.add_argument(item.arg, dest=f"item{index}")
    parser.add_argument(_RESERVED["arg"], dest="config_file")
    try:
        parsed, extras = parser.parse_known_args(argv)
    except _ArgvError as exc:
        return {}, None, [f"  the arguments ({Source('arg')}): {exc}"], True

    found = vars(parsed)
    values = {
        index: found[f"item{index}"]
        for index in range(len(items))
        if f"item{index}" in found
    }
    errors: list[str] = []
    positionals = len(after)
    at = 0
    while at < len(extras):
        token = extras[at]
        at += 1
        flag = _flag_in(token)
        if flag is None:
            positionals += 1
            continue
        errors.append(
            f"  the arguments ({Source('arg', flag)}): no item takes this flag"
        )
        if "=" not in token and at < len(extras) and _flag_in(extras[at]) is None:
            at += 1  # its value, never shown
    if positionals:
        errors.append(_POSITIONAL.format(positionals))
    return values, found.get("config_file"), errors, False


def _read_file(path: str) -> tuple[_Layer | None, str | None]:
    """The file layer read from ``path``, or the one line saying why it can't be.

    The path is used as given. The line names the path and the trouble, never the
    file's text nor the parser's message, which quotes it.
    """
    file = Path(path)
    data: Any = None
    text: str | None = None
    trouble: str | None = None
    try:
        if not file.exists():
            trouble = "does not exist"
        elif not file.is_file():
            trouble = "is not a file"
        else:
            text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        trouble = f"cannot be read: {type(exc).__name__}"
    if text is not None:
        try:
            data = yaml.safe_load(text)
        # Whatever the loader raises: a constructor's ValueError, such as a bad date,
        # quotes the value, and a bad tag's value can end in an AttributeError.
        except Exception as exc:  # noqa: BLE001  # CFG-9: never the exception's text
            mark = getattr(exc, "problem_mark", None)
            trouble = "is not valid YAML"
            if mark is not None:
                trouble += f" at line {mark.line + 1}, column {mark.column + 1}"
    if trouble is None:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            trouble = f"must be a mapping, not a {type(data).__name__}"
        else:
            try:
                data = _unblank(data)
            except RecursionError:  # an anchor whose mapping holds itself
                trouble = "is not valid YAML"
    if trouble is not None:
        return None, f"  config file {path} {trouble}"
    return _Layer("file", data, (((), path),)), None


# Error types whose pydantic text can carry the input, and the fixed text said instead.
_FIXED_MESSAGES = {
    "union_tag_invalid": "the tag isn't one of the expected values",
    "zoneinfo_str": "invalid timezone",
    "byte_size_unit": "could not interpret the byte unit",
    "timezone_offset": "the timezone offset isn't the expected one",
    "bytes_invalid_encoding": "the data isn't valid in the expected encoding",
}
# A server's own validator raises these with its exception in ``ctx["error"]``.
# Pydantic's own checks use ``value_error`` too, with a string detail, so the context
# decides.
_OWN_WORDS = frozenset({"value_error", "assertion_error"})


def _message(error: Mapping[str, Any]) -> str:
    """The text for one validation error, never carrying the input.

    Pydantic's text is kept when it comes from the schema alone. The ones that append
    what went wrong with the input (``*_parsing``, ``json_invalid``, ``import_error``)
    are cut at their first ``, `` or ``: ``, which always comes before the detail, and a
    few others are replaced by a fixed text.
    """
    kind = error["type"]
    ctx = error.get("ctx") or {}
    if kind in _OWN_WORDS and isinstance(ctx.get("error"), Exception):
        return error["msg"]
    if kind in _FIXED_MESSAGES:
        return _FIXED_MESSAGES[kind]
    msg = error["msg"]
    if "error" in ctx or "reason" in ctx:
        cuts = [i for i in (msg.find(", "), msg.find(": ")) if i >= 0]
        return msg[: min(cuts)] if cuts else msg
    return msg


_BOTH = (
    "config= reads no flags or env vars: pass config=, or argv= and environ=, not both"
)
_BAD_ARGV = "argv= takes a list of strings, such as sys.argv[1:]"
_BAD_ENVIRON = "environ= takes a mapping of env var names to values, such as os.environ"


def _given(
    composition: _Composition,
    values: Mapping[str, Any] | BaseModel | Iterable[BaseModel],
) -> dict[str, Any]:
    """Values given in code as one mapping: as given, or section instances spread."""
    if isinstance(values, BaseModel):
        return _spread([values], composition.held, composition.replaced)
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, type):
        raise _bad_values(f"the class {values.__qualname__}")
    if isinstance(values, (str, bytes)):
        raise _bad_values(f"an instance of {type(values).__qualname__}")
    if isinstance(values, Iterable):
        return _spread(values, composition.held, composition.replaced)
    raise _bad_values(f"an instance of {type(values).__qualname__}")


_ANY = TypeAdapter(Any)


def _type_name(value: object) -> str:
    return f"<{type(value).__qualname__}>"


def _json(value: object) -> str:
    """A loaded value as JSON for the boot line, or its type's name where it has none.

    A field typed ``Any`` may hold an object pydantic can't write, and a load that
    validated must not fail on its log line, so such a value is named by its type and
    never shown.
    """
    try:
        return _ANY.dump_json(value, fallback=_type_name).decode()
    except ValueError:  # pydantic's serialization error, such as bytes not UTF-8
        return json.dumps(_type_name(value))


_MASK = "**********"


def _mask(value: object) -> object:
    if isinstance(value, (SecretStr, SecretBytes, Secret)):
        return _MASK
    if isinstance(value, RootModel):
        return _mask(value.root)
    if isinstance(value, BaseModel):
        fields = {name: getattr(value, name) for name in type(value).model_fields}
        return {k: _mask(v) for k, v in {**fields, **(value.model_extra or {})}.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _mask(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_mask(v) for v in value]
    return value


def _masked(value: object) -> object:
    """``value`` with every secret in it written as ``**********``, wherever it sits.

    A model and a dataclass become a dict of its fields by attribute name, so its own
    serializers never run. A mapping's values and a list's, tuple's or set's items are
    masked in turn. A value nested too deep to walk is named by its type.
    """
    try:
        return _mask(value)
    except RecursionError:
        return _type_name(value)


def _boot_line(
    leaves: Iterable[_Item], loaded: Config, sources: Mapping[str, Source]
) -> str:
    """The one line a load logs: each leaf not at its default, its value and source.

    A sensitive item's value is ``***``. Every other value is the loaded one, as JSON,
    masked at runtime before it is written, so no serializer of the server's can print
    a secret.
    """
    shown: list[str] = []
    for item in leaves:
        source = sources[item.path]
        if source.kind == "default":
            continue
        if item.sensitive:
            value = "***"
        else:
            node: Any = loaded
            for attr in item.attrs:
                node = getattr(node, attr, None)
            value = _json(_masked(node))
        shown.append(f"{item.path}={value} ({source})")
    return "config: " + (", ".join(shown) if shown else "all defaults")


def _drop(data: Mapping[str, Any], loc: tuple[str, ...]) -> dict[str, Any]:
    """``data`` without the value at ``loc``, copying the mappings on the way down."""
    head, rest = loc[0], loc[1:]
    out = dict(data)
    if rest:
        out[head] = _drop(out[head], rest)
    else:
        del out[head]
    return out


def _check_shapes(
    items: Sequence[_Item], layers: Sequence[_Layer]
) -> tuple[list[_Layer], list[str]]:
    """The layers without a non-mapping where a model belongs, and an error line each.

    Each non-leaf item's ``loc`` is checked in every layer: a ``BaseModel`` instance
    or a mapping there is fine, and ``None`` is left to the validators. Anything else
    is dropped from that layer, so the validators don't report the path again, and
    named by its type and its layer's source. Items go depth-first, so a parent's
    error removes what its children would see.
    """
    errors: list[str] = []
    checked = list(layers)
    for item in items:
        if item.leaf or not item.loc:
            continue
        for index, layer in enumerate(checked):
            node: Any = layer.data
            for part in item.loc:
                if not isinstance(node, Mapping) or part not in node:
                    node = None
                    break
                node = node[part]
            else:
                if node is None or isinstance(node, (Mapping, BaseModel)):
                    continue
                errors.append(
                    f"  {item.path} ({_source(item.loc, [layer])}): "
                    f"must be a mapping, not a {type(node).__name__}"
                )
                checked[index] = layer._replace(data=_drop(layer.data, item.loc))
    return checked, errors


_BAD_HOOKS = "hooks= takes a mapping of built-in keys to callables"


def _check_hook_mapping(
    hooks: Mapping[str, Callable[[Config], BaseModel]] | None,
) -> dict[str, Callable[[Config], BaseModel]]:
    """``hooks=`` as a dict, each key a built-in's and each value callable."""
    if hooks is None:
        return {}
    if not isinstance(hooks, Mapping):
        raise TypeError(_BAD_HOOKS)
    _check_hooks(hooks.keys(), "hooks=")
    for key, hook in hooks.items():
        if not callable(hook):
            raise TypeError(f"hooks[{key!r}] must be callable")
    return dict(hooks)


def _without(layer: _Layer, hooked: AbstractSet[str]) -> _Layer:
    """A layer without its entries under the hooked keys."""
    if not hooked:
        return layer
    return layer._replace(data={k: v for k, v in layer.data.items() if k not in hooked})


def _run_hooks(
    composition: _Composition,
    hooks: Mapping[str, Callable[[Config], BaseModel]],
    loaded: Config,
) -> tuple[dict[str, BaseModel], list[str]]:
    """Each hook's section, validated as its own class, and the errors of those that
    don't validate.

    The hooks run in built-in order, each given ``loaded``. A hook that raises
    propagates as it is. A field a result doesn't hold, as ``model_construct`` can
    leave one, is ``Field required``, whatever its default.
    """
    validated: dict[str, BaseModel] = {}
    lines: list[str] = []
    for key in (k for k in _builtin_keys() if k in hooks):
        holder = composition.held[key]
        result = hooks[key](loaded)
        if not isinstance(result, holder):
            raise TypeError(
                f"hooks[{key!r}] returned a {type(result).__qualname__}, "
                f"not an instance of {holder.__qualname__}"
            )
        cls = type(result)
        missing = [
            f"  {key}.{_path((_key(cls, name, info),), cls)} ({Source('hook')}): "
            "Field required"
            for name, info in cls.model_fields.items()
            if name not in vars(result)
        ]
        if missing:
            lines += missing
            continue
        try:
            validated[key] = cls.model_validate(_merge(result, {}))
        except ValidationError as exc:
            lines += [
                f"  {key}"
                + (f".{_path(error['loc'], cls)}" if error["loc"] else "")
                + f" ({Source('hook')}): {_message(error)}"
                for error in exc.errors(include_input=False, include_url=False)
            ]
    return validated, lines


def _validate(
    composition: _Composition, data: Mapping[str, Any], layers: Sequence[_Layer]
) -> tuple[Config | None, list[str]]:
    """The composed model validated from ``data``, or the error line of each failure."""
    try:
        return composition.model.model_validate(data), []
    except ValidationError as exc:
        return None, [
            f"  {_path(error['loc'], composition.model)} "
            f"({_source(error['loc'], layers)}): "
            f"{_message(error)}"
            for error in exc.errors(include_input=False, include_url=False)
        ]


def _load(
    sources: Iterable[Iterable[type[BaseModel]]],
    values: Mapping[str, Any] | BaseModel | Iterable[BaseModel] | None,
    *,
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    hooks: Mapping[str, Callable[[Config], BaseModel]] | None = None,
) -> _Loaded:
    """Compose the sections and load one config.

    Without ``values``, each source goes over the one below: the defaults, then the
    YAML file named by ``--config-file`` or ``CONFIG_FILE``, then each item's env var,
    then each item's flag. Mappings merge key by key at every depth, and anything else
    replaces what is below it whole. A blank value is unset. ``argv=None`` reads no
    flags, as nothing reads ``sys.argv``, and ``environ=None`` reads ``os.environ``.

    ``values`` is a mapping, one section instance, or an iterable of them, given in
    code: it goes over the defaults, and no file, env var or flag is read. Instances
    given are kept as given.

    Each leaf item's path maps, in item order and read-only, to the source its value
    came from. A load that validates builds its boot line, which names each item not at
    its default, its value and its source, with a sensitive item's value as ``***``.
    The load logs nothing: the server logs the line each time serving starts.

    A load error names each path and its source, and a path stops at a sensitive
    field. Pydantic's own texts never carry the input, but a server's own validator
    text is kept as written, so a validator should not put the value in it. The argv
    errors, the decode errors and every validation error are reported together. A
    named file that can't be used stops the load at once, alone.

    A root instance's fields are spread into the mapping and validated again with the
    rest, as are the values of a default model instance a source merges over, so a
    validator that isn't idempotent runs twice on them.

    ``hooks`` maps a built-in's key, ``serving``, ``telemetry`` or ``delivery``, to a
    callable that supplies that section from the server's own config. A hooked key is
    left out of every source: its env vars aren't read, its flags aren't parsed, so
    passing one is an unknown flag, and the entries under it in the file or in
    ``values`` are dropped. It loads at its defaults, so a hooked section's every field
    needs a default, since its own sources aren't read, and its names aren't checked
    against the server's. Once the config validates, the hooks run in built-in order,
    each given that config, and each must return an instance of the class holding the
    key. Each result is validated again as its own class, its errors reported together
    under ``<key>.<path> (hook)``, and then the whole config is validated again with
    each validated section in place, so a root validator sees the hooked values and runs
    twice. Every leaf under a hooked key has the source ``hook``. A hook that raises
    propagates as it is.

    Raises:
        TypeError: ``values`` isn't a mapping or section instances, ``values`` is
            given with ``argv`` or ``environ``, ``argv`` isn't a sequence of ``str``,
            ``environ`` isn't a mapping, ``hooks`` isn't a mapping of callables, or a
            hook returns something other than an instance of its key's class.
        ValueError: Two instances for one key, a key of ``hooks`` isn't a built-in's,
            or the config doesn't load.
    """
    if values is not None and (argv is not None or environ is not None):
        raise TypeError(_BOTH)
    if argv is not None and (
        isinstance(argv, (str, bytes, bytearray))
        or not isinstance(argv, Sequence)
        or not all(isinstance(token, str) for token in argv)
    ):
        raise TypeError(_BAD_ARGV)
    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError(_BAD_ENVIRON)
    hooks = _check_hook_mapping(hooks)
    hooked = frozenset(hooks)

    composition = _compose(sources, hooked=hooked)
    items = composition.items
    # The items whose names are read: one list for the env vars and the flags, so the
    # flags' indexes are into it.
    read = tuple(item for item in items if not _is_hooked(item, hooked))
    lines: list[str] = []
    layers: list[_Layer] = []
    if values is not None:
        code = _Layer("code", _given(composition, values), (((), None),))
        layers.append(_without(code, hooked))
    else:
        environ = os.environ if environ is None else environ
        flags: dict[int, str] = {}
        path: str | None = None
        failed = False
        if argv is not None:
            flags, path, lines, failed = _read_argv(read, argv)
        if not failed:
            if _blank(path):
                path = environ.get(_RESERVED["env"])
            if not _blank(path):
                file, trouble = _read_file(path)
                if trouble is not None:
                    raise ValueError("the config did not load:\n" + trouble)
                layers.append(_without(file, hooked))
        env, env_errors = _read_env(read, environ)
        arg, arg_errors = _named_layer(
            "arg",
            (
                (index, read[index], read[index].arg, raw)
                for index, raw in sorted(flags.items())
            ),
        )
        lines += env_errors + arg_errors
        layers += [layer for layer in (env, arg) if layer.names]

    layers, shape_errors = _check_shapes(items, layers)
    lines += shape_errors
    data: dict[str, Any] = {}
    for layer in layers:
        data = _merge(data, layer.data)
    data = _over_defaults(composition.model, data)
    # Built outside an ``except``, so no original error is on a chain.
    loaded, errors = _validate(composition, data, layers)
    lines += errors
    if not lines and hooks:
        assert loaded is not None
        validated, lines = _run_hooks(composition, hooks, loaded)
        if not lines:
            layers.append(
                _Layer("hook", validated, tuple(((key,), None) for key in validated))
            )
            loaded, lines = _validate(composition, {**data, **validated}, layers)
    if lines:
        raise ValueError("the config did not load:\n" + "\n".join(lines))
    assert loaded is not None

    provided: dict[type, object] = {Config: loaded}
    for key, cls in composition.held.items():
        value = loaded if key is None else getattr(loaded, key)
        for each in (cls, *composition.replaced.get(key, ())):
            provided[each] = value
    leaves = [item for item in items if item.leaf]
    record = types.MappingProxyType(
        {item.path: _source(item.loc, layers) for item in leaves}
    )
    return _Loaded(loaded, provided, record, _boot_line(leaves, loaded, record))


class LoadedConfig:
    """A server's loaded config, as ``load_config()`` returns it.

    Made by ``load_config()``, never directly, and passed to ``QuickMCP(name,
    loaded=...)``. ``config`` is the validated config and ``sources`` says where each
    of its values came from, by dotted path. Both are read-only.
    """

    __slots__ = ("_components", "_loaded")

    def __init__(self, loaded: _Loaded, components: tuple[Component, ...]) -> None:
        self._loaded = loaded
        self._components = components

    @property
    def config(self) -> Config:
        """The loaded config: the root, with each section under its key."""
        return self._loaded.config

    @property
    def sources(self) -> Mapping[str, Source]:
        """Where each value came from, by dotted path. Read-only."""
        return self._loaded.sources


def load_config(
    *,
    sections: Iterable[type[BaseModel]] = (),
    components: Iterable[Component] = (),
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | BaseModel | Iterable[BaseModel] | None = None,
    hooks: Mapping[str, Callable[[Config], BaseModel]] | None = None,
) -> LoadedConfig:
    """Loads a server's config before the server is built.

    Loads as ``QuickMCP(...)`` does with the same arguments, the built-in sections
    included, and raises what it raises, with the same text. It logs nothing and
    imports no FastMCP. Its result goes to ``QuickMCP(name, loaded=...)``, which
    reads no source again.
    """
    checked = _components(components)
    loaded = _load(
        [[c.section for c in checked if c.section is not None], sections],
        config,
        argv=argv,
        environ=environ,
        hooks=hooks,
    )
    return LoadedConfig(loaded, checked)


# Schema keywords whose value maps a name to a schema, so a name there is data.
_NAME_MAPS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
# Keywords whose value is data, not a schema: nothing in it is a keyword.
_DATA = frozenset({"default", "examples", "enum", "const"})


def _holds_object(value: object) -> bool:
    """Whether a JSON value is, or holds anywhere inside it, an object."""
    if isinstance(value, dict):
        return True
    return isinstance(value, list) and any(_holds_object(item) for item in value)


def _reaches_model(node: object, *, names: bool = False) -> bool:
    """Whether a schema, below its data, refers to another schema through a ``$ref``."""
    if isinstance(node, list):
        return any(_reaches_model(item) for item in node)
    if not isinstance(node, dict):
        return False
    if not names and "$ref" in node:
        return True
    return any(
        _reaches_model(value, names=not names and key in _NAME_MAPS)
        for key, value in node.items()
        if names or key not in _DATA
    )


def _scrub(node: object, *, names: bool = False) -> None:
    """Remove, in place, every default that could print a secret.

    That is the default of a sensitive field, with its examples, and any default that
    holds a JSON object or sits on a field whose type is a model, since a model instance
    written as a default may carry a sensitive item, and a serializer may write it as
    anything. ``names`` marks a map from names to schemas, whose keys aren't keywords.
    """
    if isinstance(node, list):
        for item in node:
            _scrub(item)
    elif isinstance(node, dict):
        if not names:
            if node.get("sensitive") is True:
                node.pop("default", None)
                node.pop("examples", None)
            elif "default" in node and (
                _holds_object(node["default"]) or _reaches_model(node)
            ):
                del node["default"]
        for key, value in node.items():
            if names or key not in _DATA:
                _scrub(value, names=not names and key in _NAME_MAPS)


class _Schema(GenerateJsonSchema):
    """Pydantic's schema, with ``"sensitive": true`` on every field of a secret type.

    Pydantic marks ``SecretStr`` and ``SecretBytes`` as ``writeOnly``, but not
    ``Secret[T]``, so the mark is written from each model's fields rather than read from
    pydantic's output. ``_scrub`` then drops every marked property's default and
    examples. The properties of each class in ``unread`` lose their ``env`` and
    ``arg``, which a hooked section's items don't read.
    """

    unread: typing.ClassVar[frozenset[type]] = frozenset()

    def model_schema(self, schema: Any) -> Any:
        json_schema = super().model_schema(schema)
        cls = schema["cls"]
        properties = json_schema.get("properties")
        if not isinstance(properties, dict):
            return json_schema
        for name in _sensitive_fields(cls):
            prop = properties.get(_schema_key(name, cls.model_fields[name]))
            if isinstance(prop, dict):
                prop["sensitive"] = True
        if cls in self.unread:
            for prop in properties.values():
                if isinstance(prop, dict):
                    prop.pop("env", None)
                    prop.pop("arg", None)
        return json_schema


def _reached(classes: Iterable[type[BaseModel]]) -> set[type[BaseModel]]:
    """The classes and every model their fields reach, at any depth."""
    queue = list(classes)
    seen = set(queue)
    while queue:
        cls = queue.pop()
        for info in cls.model_fields.values():
            for model in _models_in(info.annotation):
                if issubclass(model, BaseModel) and model not in seen:
                    seen.add(model)
                    queue.append(model)
    return seen


def _unread(composition: _Composition, hooked: AbstractSet[str]) -> frozenset[type]:
    """The models reached only through the hooked sections, whose names aren't read.

    A model a section that isn't hooked reaches too keeps its names, which are read
    there: the schema has one definition per class.
    """
    held = composition.held
    under = _reached(held[key] for key in hooked)
    read = _reached(cls for key, cls in held.items() if key not in hooked)
    return frozenset(under - read)


_BAD_HOOKED = "hooked= takes an iterable of built-in keys, such as ['serving']"


def config_schema(
    *sections: type[BaseModel], hooked: Iterable[str] = ()
) -> dict[str, Any]:
    """The JSON Schema of the config that ``sections`` compose, without a server.

    The built-in sections are composed first and every registration rule applies, so a
    bad section raises what ``QuickMCP(sections=...)`` would. ``Field(doc=)`` appears as
    ``description``, ``env`` and ``arg`` as written, every field of a secret type as
    ``sensitive: true``, and a section that forbids unknown keys as
    ``additionalProperties: false``. The default of a sensitive field is left out, and
    so is any default that is a JSON object or sits on a model-typed field, which could
    hold one.

    ``hooked`` names the built-ins a server supplies through ``hooks=``, and composes as
    that server does: those sections stay in the schema, but their properties, at any
    depth, have no ``env`` or ``arg``, which aren't read, and their names aren't checked
    against the others. A model that a section not hooked reaches too keeps its names.

    Raises:
        TypeError: A section isn't a marked, strict pydantic model, an item's name or
            ``sensitive=True`` can't be used, or ``hooked`` isn't an iterable of keys.
        ValueError: Two sections claim one key, a key clashes with the root, two items
            declare one name, or a key of ``hooked`` isn't a built-in's.
    """
    if isinstance(hooked, (str, bytes)) or not isinstance(hooked, Iterable):
        raise TypeError(_BAD_HOOKED)
    keys = _check_hooks(hooked, "hooked=")
    composition = _compose([sections], hooked=keys)
    generator = type(
        "_HookedSchema", (_Schema,), {"unread": _unread(composition, keys)}
    )
    with warnings.catch_warnings():
        # Pydantic's warning about a default it can't write prints the default, and
        # the scrub leaves every such default out of the schema anyway.
        warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
        schema = composition.model.model_json_schema(schema_generator=generator)
    _scrub(schema)
    return schema
