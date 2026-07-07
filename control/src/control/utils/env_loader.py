import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv


def _resolve_env_path() -> Path | None:
    """Return the .env file that will be (or was) loaded, or None if not found."""
    env_file: Optional[str] = os.getenv("PSETI_ENV_FILE")
    if env_file:
        p = Path(env_file)
        return p if p.is_file() else None
    default = Path(".env")
    return default if default.is_file() else None


def load_pseti_env() -> None:
    """
    Loads environment variables from a .env file into os.environ.

    If the 'PSETI_ENV_FILE' environment variable is set, it will attempt to load
    that specific file. Otherwise, it will look for a '.env' file in the current
    working directory.

    Variables loaded from the .env file will overwrite any existing variables in
    os.environ, to allow for flexible dynamic reconfiguration (override=True).
    """
    env_path = _resolve_env_path()
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)


def get_env_info() -> dict:
    """
    Return a snapshot of the pseti environment for display purposes.

    Returns a dict with:
        env_file: Path | None — the .env file that was/will be loaded
        dotenv_vars: dict[str, str | None] — key/value pairs parsed from that file
        pseti_vars: dict[str, str | None] — all PSETI_* keys in os.environ
        runtime_vars: dict[str, str | None] — other known pseti runtime vars from os.environ
    """
    known_runtime = [
        "DAQ_DATA_GATEWAY_HOST",
        "DAQ_DATA_GATEWAY_PORT",
        "GRPC_PORT",
        "REDIS_HOST",
        "LOKI_URL",
        "PSETI_ENV_FILE",
        "PSETI_STRICT",
        "PSETI_TEST_TIER",
        "HEAD_DATA_DIR",
    ]

    env_path = _resolve_env_path()
    dotenv_vars: dict[str, str | None] = dict(dotenv_values(env_path)) if env_path else {}

    pseti_vars = {k: v for k, v in os.environ.items() if k.startswith("PSETI_")}
    runtime_vars = {k: os.environ.get(k) for k in known_runtime if os.environ.get(k) is not None}

    return {
        "env_file": env_path,
        "dotenv_vars": dotenv_vars,
        "pseti_vars": pseti_vars,
        "runtime_vars": runtime_vars,
    }
