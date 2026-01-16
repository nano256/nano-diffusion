from pathlib import Path

import ipywidgets as widgets
import mlflow


def create_mlflow_run_selector(mlruns_path=None):
    """Create interactive widgets to select MLflow experiment and run.

    Args:
        mlruns_path: Path to mlruns directory. Defaults to ../mlruns from notebook.

    Returns:
        tuple: (exp_selector, run_selector) widget objects

    Usage:
        exp_sel, run_sel = create_mlflow_run_selector()
        display(exp_sel, run_sel)

        # In next cell:
        run = mlflow.get_run(run_sel.value)
    """
    if mlruns_path is None:
        mlruns_path = Path.cwd().parent / "mlruns"

    mlflow.set_tracking_uri("file://" + str(mlruns_path))

    experiments = {
        exp.name: exp.experiment_id
        for exp in mlflow.search_experiments()
        if exp.name != "Default"
    }

    exp_selector = widgets.Dropdown(
        options=list(experiments.keys()),
        description="Experiment:",
        layout=widgets.Layout(width="500px"),
    )

    run_selector = widgets.Dropdown(
        options=[], description="Run:", layout=widgets.Layout(width="500px")
    )

    # Update runs when experiment changes
    def update_runs(change):
        runs = mlflow.search_runs(
            experiment_ids=[experiments[change["new"]]], output_format="list"
        )
        run_selector.options = {run.info.run_name: run.info.run_id for run in runs}

    exp_selector.observe(update_runs, names="value")

    if experiments:
        update_runs({"new": exp_selector.value})

    return exp_selector, run_selector
