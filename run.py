# This script acts as an entrypoint and is used to setup, manage and run conda environments.

import sys
from pathlib import Path
from validator import conda

ENV_NAME_PREFIX = "flock-validation-"


def entrypoint():
    repo_path = Path(__file__).resolve().parent.parent
    # is_latest_version(str(repo_path))

    module = sys.argv[1]
    # Check if the module directory exists
    base_requirements_file = Path(__file__).parent / "requirements.txt"
    module_environment_file = Path(__file__).parent / "validator" / "modules" / module / "environment.yml"
    if not module_environment_file.exists():
        raise ValueError(f"Module {module} does not exist")
    
    # Check if the environment exists
    env_name = ENV_NAME_PREFIX + module

    # Run the module, passing through all the arguments
    conda.ensure_env_and_run(env_name, module_environment_file, base_requirements_file, ["--no-capture-output", "python", "environment_entrypoint.py", *sys.argv[1:]])


if __name__ == "__main__":
    entrypoint()