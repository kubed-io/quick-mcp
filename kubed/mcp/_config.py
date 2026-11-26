"""One config per server, composed from the sections the server registers.

A server's config is not one big class. It is a root section plus keyed sections, each
a pydantic model marked with ``@config``, and this module composes them into one strict
model. It imports no FastMCP, so a config can be loaded and described without a server.
"""

from __future__ import annotations

import dataclasses
import keyword
import re
import types
import typing
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Annotated, Any, Literal, NamedTuple, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    Secret,
    SecretBytes,
    SecretStr,
    ValidationError,
    create_model,
)
from pydantic import Field as _PydanticField
from pydantic.errors import PydanticUndefinedAnnotation, PydanticUserError
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, PydanticJsonSchemaWarning

M = TypeVar("M", bound=BaseModel)

# The stamp ``config`` leaves on a class. A dunder name, so pydantic leaves it alone,
# and read from the class's own ``vars()``, so a subclass never inherits a key.
_STAMP = "__kubed_config_key__"

# "No default given" for ``Field``, so ``None`` stays a usable default.
_UNSET: Any = object()

# The sections every server has. Empty until a built-in section exists.
_BUILTINS: tuple[type[BaseModel], ...] = ()


class Config(BaseModel):
    """The base a section may subclass. It refuses unknown keys.

    ``Inject[Config]`` resolves to the whole loaded config.
    """

    model_config = ConfigDict(extra="forbid")


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
    that set it, as written, ``sensitive`` says a secret type is in its annotation, and
    ``takes`` says how a raw value from an env var or a flag decodes.
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

    A NewType and a type alias are followed to what they name. ``Annotated`` metadata,
    such as a nested ``FieldInfo``, is a node too.
    """
    if hasattr(annotation, "__supertype__"):  # a NewType
        yield from _parts(annotation.__supertype__)
        return
    if hasattr(annotation, "__value__") and not isinstance(annotation, type):
        yield from _parts(annotation.__value__)  # a TypeAliasType
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


# The keywords ``Field`` writes that only mean something on the field itself.
_ITEM_KEYWORDS = ("env", "arg", "sensitive")


def _check_item_keywords(cls: type[BaseModel], name: str, path: str) -> None:
    """Refuse item keywords that would silently do nothing, or a false ``sensitive``."""
    info = cls.model_fields[name]
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
    """How a field's raw value decodes. ``NoDecode`` anywhere on it makes it text."""
    marks = (*info.metadata, *_parts(info.annotation))
    if any(m is NoDecode or isinstance(m, NoDecode) for m in marks):
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
        alias = info.validation_alias
        here = (*loc, alias if isinstance(alias, str) else name)
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
                sensitive=name in _sensitive_fields(cls),
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


def _check_names(items: Iterable[_Item]) -> None:
    """Refuse a malformed name, one of quick-mcp's own, or one declared twice."""
    declared: dict[tuple[str, str], str] = {}
    for item in items:
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
            first = declared.setdefault((kind, name), item.path)
            if first != item.path:
                raise ValueError(
                    f"The {_NAME_KIND[kind]} {name!r} is declared by both {first} "
                    f"and {item.path}"
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


def _items(model: type[BaseModel], root: type[BaseModel] | None) -> tuple[_Item, ...]:
    """The composed model's items, depth first in declaration order, checked."""
    items: list[_Item] = []
    direct: set[type] = {model} if root is None else {model, root}
    held_in: dict[type, str] = {}
    _walk_items(model, (), (), frozenset({model}), items, direct, held_in)
    _check_names(items)
    _check_held(held_in, direct)
    return tuple(items)


_NOT_ITERABLE = "sections= takes an iterable of section classes, such as a list"


def _compose(sources: Iterable[Iterable[type[BaseModel]]]) -> _Composition:
    """Register the sections of every source and compose them into one config model.

    ``_BUILTINS`` come first, then each source in order. The most derived class wins
    a key. Every field reached through model fields is recorded as an item, and the
    names items declare are checked here, so ``config_schema`` refuses what a server
    refuses.
    """
    held: dict[str | None, type[BaseModel]] = {}
    replaced: dict[str | None, list[type[BaseModel]]] = {}
    if not isinstance(sources, Iterable):
        raise TypeError(_NOT_ITERABLE)
    for source in (_BUILTINS, *sources):
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
        _items(model, root),
    )


class _Loaded(NamedTuple):
    """What ``_load`` returns: the config, and what to inject for each class."""

    config: Config
    provided: dict[type, object]


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


def _path(loc: tuple[int | str, ...]) -> str:
    """Join a pydantic ``loc``: names with ``.``, indexes as ``[i]``."""
    if not loc:
        return "the root section"
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif part.startswith("["):  # pydantic's "[key]", for a bad dict key
            path += part
        else:
            path += f".{part}" if path else part
    return path


def _source(loc: tuple[int | str, ...], given: Mapping[str, Any] | None) -> str:
    """Where a value at ``loc`` came from. Files, env vars and argv join this later."""
    if given is None:
        return "default"
    if not loc or loc[0] in given:
        return "code"
    return "default"


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


def _load(
    sources: Iterable[Iterable[type[BaseModel]]],
    values: Mapping[str, Any] | BaseModel | Iterable[BaseModel] | None,
) -> _Loaded:
    """Compose the sections and load one config from defaults and values given in code.

    ``values`` is a mapping, one section instance, or an iterable of them. Given values
    are never read from a file, an env var or argv. A load error names each path and
    its source. Pydantic's own texts never carry the input, but a server's own validator
    text is kept as written, so a validator should not put the value in it.

    A root instance's fields are spread into the mapping and validated again with the
    rest, so a root validator that isn't idempotent runs twice on them.

    Raises:
        TypeError: ``values`` isn't a mapping or section instances.
        ValueError: Two instances for one key, or the values don't validate.
    """
    composition = _compose(sources)
    given: dict[str, Any] | None
    if values is None:
        given = None
    elif isinstance(values, BaseModel):
        given = _spread([values], composition.held, composition.replaced)
    elif isinstance(values, Mapping):
        given = dict(values)
    elif isinstance(values, type):
        raise _bad_values(f"the class {values.__qualname__}")
    elif isinstance(values, (str, bytes)):
        raise _bad_values(f"an instance of {type(values).__qualname__}")
    elif isinstance(values, Iterable):
        given = _spread(values, composition.held, composition.replaced)
    else:
        raise _bad_values(f"an instance of {type(values).__qualname__}")

    lines: list[str] = []
    try:
        loaded = composition.model.model_validate({} if given is None else given)
    except ValidationError as exc:
        lines = [
            f"  {_path(error['loc'])} ({_source(error['loc'], given)}): "
            f"{_message(error)}"
            for error in exc.errors(include_input=False, include_url=False)
        ]
    if lines:
        # Raised outside the ``except``, so the original error is on no chain.
        raise ValueError("the config did not load:\n" + "\n".join(lines))

    provided: dict[type, object] = {Config: loaded}
    for key, cls in composition.held.items():
        value = loaded if key is None else getattr(loaded, key)
        for each in (cls, *composition.replaced.get(key, ())):
            provided[each] = value
    return _Loaded(loaded, provided)


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
    examples.
    """

    def model_schema(self, schema: Any) -> Any:
        json_schema = super().model_schema(schema)
        cls = schema["cls"]
        properties = json_schema.get("properties")
        if not isinstance(properties, dict):
            return json_schema
        for name in _sensitive_fields(cls):
            alias = cls.model_fields[name].validation_alias
            prop = properties.get(alias if isinstance(alias, str) else name)
            if isinstance(prop, dict):
                prop["sensitive"] = True
        return json_schema


def config_schema(*sections: type[BaseModel]) -> dict[str, Any]:
    """The JSON Schema of the config that ``sections`` compose, without a server.

    The built-in sections are composed first and every registration rule applies, so a
    bad section raises what ``QuickMCP(sections=...)`` would. ``Field(doc=)`` appears as
    ``description``, ``env`` and ``arg`` as written, every field of a secret type as
    ``sensitive: true``, and a section that forbids unknown keys as
    ``additionalProperties: false``. The default of a sensitive field is left out, and
    so is any default that is a JSON object or sits on a model-typed field, which could
    hold one.

    Raises:
        TypeError: A section isn't a marked, strict pydantic model, or an item's name
            or ``sensitive=True`` can't be used.
        ValueError: Two sections claim one key, a key clashes with the root, or two
            items declare one name.
    """
    with warnings.catch_warnings():
        # Pydantic's warning about a default it can't write prints the default, and
        # the scrub leaves every such default out of the schema anyway.
        warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
        schema = _compose([sections]).model.model_json_schema(schema_generator=_Schema)
    _scrub(schema)
    return schema
