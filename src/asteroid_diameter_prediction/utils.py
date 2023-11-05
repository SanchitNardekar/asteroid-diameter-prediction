# src/asteroid-diameter-prediction/utils.py

import yaml
from pathlib import Path
import time
from datetime import timedelta


def get_git_root(path=Path.cwd()):
    """Function to get the git root."""
    try:
        import git

        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.working_dir
        return git_root

    except ModuleNotFoundError:
        print("Git not available, returning estimate of git root from package.")
        import asteroid_diameter_prediction

        return str(Path(asteroid_diameter_prediction.__file__).parents[2])


def get_params():
    """Load params from params.yaml."""
    with open(Path(get_git_root()) / "params.yaml") as file:
        model_args = yaml.load(file, Loader=yaml.FullLoader)
    return model_args
