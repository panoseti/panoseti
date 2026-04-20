#! /usr/bin/env python3



import control.power as power
from control.utils import config_file, util
from control.utils.pydantic_config_models import ObsConfigValidator


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
        print("You don't have permission to stop the redis daemons. "
              "Try running 'sudo ./config.py --stop_redis_daemons'.")

import typer

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
    

