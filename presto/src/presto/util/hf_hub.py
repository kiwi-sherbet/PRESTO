#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from contextlib import contextmanager


@dataclass
class HfConfig:
    use_hfhub: bool = True
    hf_repo_id: Optional[str] = 'dmp2023/presto-model'


def upload_ckpt(repo_id: str,
                ckpt_file: str,
                name: Optional[str] = None):
    ckpt_file = Path(ckpt_file)

    api = HfApi()

    # 1. Create repo.
    url = api.create_repo(repo_id,
                          repo_type=None,  # =="model"
                          exist_ok=True)

    # 2. [Optional] auto-configure name
    if name is None:
        name = ckpt_file.name

    # 3. Upload file.
    api.upload_file(
        path_or_fileobj=str(ckpt_file),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type='model'
    )


def download_ckpt(repo_id: str, name: str) -> str:
    """ Download checkpoint from huggingface model hub. """
    ckpt_file: str = hf_hub_download(repo_id, name)
    return ckpt_file


@contextmanager
def with_hfhub(ckpt_file: str,
               hf_repo_id: str = 'dmp2023/presto-model',
               name: Optional[str] = None,
               use: bool = True):
    try:
        yield
    finally:
        # NOTE(ycho): we assume that
        # at this point, `ckpt_file` exists.
        upload_ckpt(hf_repo_id,
                    ckpt_file,
                    name)
