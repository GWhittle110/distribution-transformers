"""
Main python file for running experiments. Pass the name of the config as an argument called from the command line.
"""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from importlib import import_module


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.automain
def main(_config=None, _run=None):
    """
    Run the experiment specified by the configuration passed to the command line.

    Args:
        _config: Experiment config, autofilled by sacred.
        _run: Sacred run object.

    """
    experiment = import_module("experiments.sources." + _config["module_path"])
    experiment.run(**_config, _run=_run)

