"""
Bridges the new stack's PSETI_CONFIG (and PSETI_TMP) env vars to the old
software's hardcoded, cwd-relative 'configs/...' and 'tmp/quabo_uids.json'
path convention (control/utils/config_file.py), with zero manual symlink
setup and zero duplicated config files.

The old loader can't be pointed at an arbitrary directory directly -- its
filename constants are hardcoded as 'configs/<file>.json' relative to a
`dir` argument that itself defaults to '.'. So instead of asking an
operator to hand-craft a symlink at 3am, this creates one automatically
(idempotent -- safe to call every run) under a local scratch directory,
then returns that scratch directory as the `dir` to pass into
config_file.get_*_config(dir=...).
"""

import os
import sys

_BRIDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_config_bridge")


def resolve_config_dir() -> str:
    """Return a `dir` value usable with config_file.get_*_config(dir=...).

    Reads PSETI_CONFIG from the environment (the same variable the new
    `pseti` CLI uses -- see `pseti env` / `pseti paths` on the live
    system) and symlinks it in as 'configs' under a local scratch
    directory, so the old loader's hardcoded 'configs/<file>.json' paths
    resolve to the exact same live config files the new stack reads.
    """
    config_dir = os.environ.get("PSETI_CONFIG")
    if not config_dir:
        sys.exit(
            "PSETI_CONFIG is not set. This fallback reads the *same* config "
            "files as the new pseti CLI -- run `pseti env` on the head node "
            "to find the active value, then:\n"
            "    export PSETI_CONFIG=/path/to/active/configs\n"
            "before re-running this script."
        )
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    if not os.path.isdir(config_dir):
        sys.exit(f"PSETI_CONFIG={config_dir} is not a directory.")

    os.makedirs(_BRIDGE_DIR, exist_ok=True)
    link = os.path.join(_BRIDGE_DIR, "configs")
    if os.path.islink(link) and os.readlink(link) == config_dir:
        pass  # already correct
    else:
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(config_dir, link)

    return _BRIDGE_DIR


def resolve_quabo_uids_path() -> str:
    """Find quabo_uids.json, trying PSETI_TMP first, then PSETI_CONFIG.

    The old loader hardcodes 'tmp/quabo_uids.json' (control/utils/
    config_file.py's quabo_uids_filename). On the live system this file
    is written by `pseti uids` under PSETI_TMP; some deployments instead
    keep a copy alongside the other config files. Checks both rather
    than requiring an operator to know which.
    """
    candidates = []
    if os.environ.get("PSETI_TMP"):
        candidates.append(os.path.join(os.environ["PSETI_TMP"], "quabo_uids.json"))
    if os.environ.get("PSETI_CONFIG"):
        candidates.append(os.path.join(os.environ["PSETI_CONFIG"], "quabo_uids.json"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit(
        "Could not find quabo_uids.json under PSETI_TMP or PSETI_CONFIG "
        f"(tried: {candidates}). Pass its path explicitly with --quabo-uids."
    )
