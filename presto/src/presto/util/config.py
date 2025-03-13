#!/usr/bin/env python3

from functools import wraps
from typing import (
    Callable, Union, Type,
    TypeVar, Optional
)
from dataclasses import is_dataclass
from omegaconf import OmegaConf
import inspect

D = TypeVar('D')


def with_oc_cli_v2(cls: Union[Type[D], D] = None,
                   cfg_file: Optional[str] = None):
    """Decorator for automatically adding parsed args from cli to entry
    point."""

    main = None
    if cls is None:
        # @with_cli()
        need_cls = True
    else:
        if callable(cls) and not is_dataclass(cls):
            # @with_cli
            main = cls
            need_cls = True
        else:
            # @with_cli(cls=Config, ...)
            need_cls = (cls is None)  # FIXME(ycho): always False.

    def decorator(main: Callable[[D], None]):
        # NOTE(ycho):
        # if `cls` is None, try to infer them from `main` signature.
        inner_cls = cls
        if need_cls:
            sig = inspect.signature(main)
            if len(sig.parameters) == 1:
                key = next(iter(sig.parameters))
                inner_cls = sig.parameters[key].annotation
            else:
                raise ValueError(
                    '#arg != 1 in main {}: Cannot infer param type.'
                    .format(sig))

        # NOTE(ycho): using @wraps to forward main() documentation.
        @wraps(main)
        def wrapper():
            getattr(main, '__doc__', '')
            cfg = OmegaConf.structured(inner_cls)
            if cfg_file is not None:
                cfg.merge_with(OmegaConf.load(cfg_file))
            cfg.merge_with_cli()
            cfg = OmegaConf.to_object(cfg)
            return main(cfg)
        return wrapper

    if main is not None:
        return decorator(main)
    else:
        return decorator
