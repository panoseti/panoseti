import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

def load_pseti_env() -> None:
    """
    Loads environment variables from a .env file into os.environ.
    
    If the 'PSETI_ENV_FILE' environment variable is set, it will attempt to load 
    that specific file. Otherwise, it will look for a '.env' file in the current working directory.
    
    Variables loaded from the .env file will overwrite any existing variables in os.environ,
    to allow for flexible dynamic reconfiguration (override=True).
    """
    env_file: Optional[str] = os.getenv("PSETI_ENV_FILE")
    
    if env_file:
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
    else:
        # Check current working directory for .env
        default_env = Path(".env")
        if default_env.is_file():
            load_dotenv(dotenv_path=default_env, override=True)
