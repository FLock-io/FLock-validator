import subprocess
from pathlib import Path
from typing import Optional, List
from loguru import logger

def run_command(cmd, **kwargs):
    # Ensure output is unbuffered
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs
    )
    for line in process.stdout:
        print(line, end='')  # Already includes newline
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def env_exists(env_name: str) -> bool:
    """Check if a conda environment exists."""
    logger.info(f"Checking if environment {env_name} exists")
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True, text=True, check=True
    )
    return any(line.split()[0] == env_name for line in result.stdout.splitlines() if line and not line.startswith("#"))

def create_env(env_name: str, env_yml: Path, requirements_txt: Path):
    """Create a conda environment from environment.yml or requirements.txt."""
    logger.info(f"Creating environment {env_name} from {env_yml} and {requirements_txt}")
    run_command(["conda", "env", "create", "-n", env_name, "-f", str(env_yml)])
    # Install base requirements
    run_command([
        "conda", "run", "-n", env_name, "pip", "install", "-r", str(requirements_txt)
    ])
    
def update_env(env_name: str, env_yml: Path, requirements_txt: Path):
    """Update a conda environment from environment.yml or requirements.txt."""
    logger.info(f"Updating environment {env_name} from {env_yml} and {requirements_txt}")
    run_command(["conda", "env", "update", "-n", env_name, "-f", str(env_yml)])
    run_command(["conda", "run", "-n", env_name, "pip", "install", "-r", str(requirements_txt)])
    

def install_in_env(env_name: str, packages: List[str]):
    """Install additional packages into an existing conda environment."""
    run_command(["conda", "install", "-y", "-n", env_name] + packages)

def run_in_env(env_name: str, command: List[str], env_vars: Optional[dict] = None, no_capture_output: bool = False):
    """Run a command inside a conda environment."""
    logger.info(f"Running command {command} in environment {env_name}")
    cmd = ["conda", "run"]
    if no_capture_output:
        cmd.append("--no-capture-output")
    cmd.extend(["-n", env_name] + command)
    
    try:
        run_command(cmd, env=env_vars)
    except subprocess.CalledProcessError as e:
        # If --no-capture-output flag caused the error (micromamba compatibility), retry without it
        if no_capture_output and e.returncode == 2:
            logger.warning("--no-capture-output not supported, retrying without it")
            cmd_without_flag = ["conda", "run", "-n", env_name] + command
            run_command(cmd_without_flag, env=env_vars)
        else:
            raise

def ensure_env_and_run(
    env_name: str,
    env_yml: Path,
    requirements_txt: Path,
    command: List[str],
    env_vars: Optional[dict] = None,
    no_capture_output: bool = False
):
    """Ensure the environment exists, create if needed, then run the command."""
    if not env_exists(env_name):
        create_env(env_name, env_yml, requirements_txt)
    else:
        update_env(env_name, env_yml, requirements_txt)
    run_in_env(env_name, command, env_vars, no_capture_output=no_capture_output)