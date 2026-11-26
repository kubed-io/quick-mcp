"""One config per server, composed from the sections the server registers.

A server's config is not one big class. It is a root section plus keyed sections, each
a pydantic model marked with ``@config``, and this module composes them into one strict
model. It imports no FastMCP, so a config can be loaded and described without a server.
"""

from __future__ import annotations

import dataclasses
import keyword
import typing
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict, create_model
from pydantic import Field as _PydanticField
from pydantic.errors import PydanticUndefinedAnnotation, PydanticUserError

M = TypeVar("M", bound=BaseModel)

# The stamp ``config`` leaves on a class. A dunder name, so pydantic leaves it alone,
# and read from the class's own ``vars()``, so a subclass never inherits a key.
_STAMP = "__kubed_config_key__"

# "No default given" for ``Field``, so ``None`` stays a usable default.
_UNSET: Any = object()

# The sections every server has. Empty until a built-in section exists.
_BUILTINS: tuple[type[BaseModel], ...] = ()


class Config(BaseModel):
    """The base a section may subclass. It refuses unknown keys."""

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
    them. ``sensitive=True`` marks a value that is never printed. They are written to
    the field's ``json_schema_extra``, and only the ones given. Every other keyword goes
    to pydantic.

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


class _Composition(NamedTuple):
    """What ``_compose`` returns: the model, who holds each key, and who lost one."""

    model: type[Config]
    held: dict[str | None, type[BaseModel]]
    replaced: dict[str | None, tuple[type[BaseModel], ...]]


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


def _models_in(annotation: object) -> Iterable[type]:
    """The classes an annotation reaches: itself, and the arguments of its generics."""
    if hasattr(annotation, "__supertype__"):  # a NewType
        yield from _models_in(annotation.__supertype__)
        return
    if hasattr(annotation, "__value__") and not isinstance(annotation, type):
        yield from _models_in(annotation.__value__)  # a TypeAliasType
        return
    if isinstance(annotation, type):
        yield annotation
    for arg in typing.get_args(annotation):
        yield from _models_in(arg)


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
            for inner in _models_in(info.annotation):
                _check_strict(inner, f"{prefix}{name}", seen)
    elif dataclasses.is_dataclass(cls) or _is_typed_dict(cls):
        raise TypeError(
            f"{cls.__qualname__} at {path} is a dataclass or TypedDict, which doesn't "
            "refuse unknown keys: make it a pydantic model that forbids them"
        )


def _compose(sources: Iterable[Iterable[type[BaseModel]]]) -> _Composition:
    """Register the sections of every source and compose them into one config model.

    ``_BUILTINS`` come first, then each source in order. The most derived class wins
    a key.
    """
    held: dict[str | None, type[BaseModel]] = {}
    replaced: dict[str | None, list[type[BaseModel]]] = {}
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
            fields[key] = (cls, _PydanticField(default_factory=cls))
        else:
            fields[key] = (cls, ...)

    bases = (root, _Top) if root is not None else (_Top,)
    model = create_model("Config", __base__=bases, __module__=__name__, **fields)
    return _Composition(
        model,
        held,
        {key: tuple(losers) for key, losers in replaced.items()},
    )
