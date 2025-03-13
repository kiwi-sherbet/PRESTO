#!/usr/bin/env python3

from os import PathLike
import pkg_resources
from typing import Union, Optional
from pathlib import Path
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import logging


def get_path(path: str) -> str:
    """Get resource path."""
    return pkg_resources.resource_filename('presto', path)


def ensure_directory(path: Union[str, PathLike]) -> Path:
    """Ensure that the directory structure exists."""
    path = Path(path)
    if path.is_dir():
        return path

    if path.exists():
        raise ValueError(F'path {path} exists and is not a directory.')
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise ValueError(F'Failed to create path {path}.')

    return path


class RunPath(object):
    """General path management over multiple experiment runs.

    NOTE(ycho): The intent of this class is mainly to avoid overwriting
    checkpoints and existing logs from a previous run -
    instead, we maintain a collision-free index based key
    for each experiment that we run and use them in a sub-folder structure.
    """

    @dataclass
    class Config:
        key_format: str = 'run-{:03d}'
        root: Optional[str] = '/tmp/'  # Alternatively, ~/.cache/presto/run/
        key: Optional[str] = None  # Empty string indicates auto increment.

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if (cfg.root is None or
                (isinstance(cfg.root, str) and cfg.root.lower() == 'none')):
            root = TemporaryDirectory()
        else:
            root = ensure_directory(
                Path(cfg.root).expanduser())
        self._root_path = root
        self.root = Path(
            root.name if isinstance(root, TemporaryDirectory) else root
        )

        # Resolve sub-directory key.
        key = cfg.key
        if key is None:
            key = self._resolve_key(self.root, self.cfg.key_format)
            logging.info(F'key={key}')

        self.dir = ensure_directory(self.root / key)
        logging.info(F'self.dir={self.dir}')

    def __del__(self):
        if isinstance(self._root_path, TemporaryDirectory):
            self._root_path.cleanup()

    @staticmethod
    def _resolve_key(root: str, key_fmt: str) -> str:
        """Get latest valid key according to `key_fmt`"""
        # Ensure `root` is a valid directory.
        root = Path(root)
        if not root.is_dir():
            raise ValueError(F'Arg root={root} is not a dir.')

        # NOTE(ycho): Loop through integers starting from 0.
        # Not necessarily efficient, but convenient.
        index = 0
        while True:
            key = key_fmt.format(index)
            if not (root / key).exists():
                break
            index += 1
        return key

    def __getattr__(self, key: str):
        """
        Convenient shorthand for fetching valid subdirectories.
        """
        return ensure_directory(self.dir / key)
