#! /usr/bin/env python3



import typer
from panoseti_grpc.telemetry.logger import get_logger

import control.power as power
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import ObsConfigValidator

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PANOSETI.SessionStop", log_dir=str(log_dir), grpc_enabled=True)

def session_stop(obs_config: ObsConfigValidator) -> None:
    """Gracefully terminate an observing session.
    
    Powers off all modules and stops background Redis daemons.

    Args:
        obs_config: Validated observatory configuration.
    """
    power.do_all(obs_config, 'off')
    try:
        util.stop_redis_daemons()
    except PermissionError:
        logger.error("You don't have permission to stop the redis daemons. "
                     "Try running 'sudo ./config.py --stop_redis_daemons'.")

app = typer.Typer(help="Gracefully terminate an observing session.", no_args_is_help=False, context_settings={"help_option_names": ["-h", "--help"]})

@app.command()
def main() -> None:
    """
    Gracefully terminate an observing session.
    
    Powers off all modules and stops background Redis daemons.
    """
    obs_config = config_file.get_obs_config()
    session_stop(obs_config)


if __name__ == "__main__":
    app()
    

