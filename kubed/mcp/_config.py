"""One config per server, composed from the sections the server registers.

A server's config is not one big class. It is a root section plus keyed sections, each
a pydantic model marked with ``@config``, and this module composes them into one strict
model. It imports no FastMCP, so a config can be loaded and described without a server.
"""

from __future__ import annotations

import dataclasses
import keyword
import typing
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
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
    its source, and never a value.

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
