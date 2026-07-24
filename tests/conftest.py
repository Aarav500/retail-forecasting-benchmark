"""Force a headless matplotlib backend for the whole test session.

Must run before any test module is collected. `shortseq.visualization.figures`
has its own `matplotlib.use("Agg")` guard, but it deliberately does NOT
override a backend that's already been resolved (see the comment there) --
and in this environment, `matplotlib.rcParams["backend"]` defaults to
"tkagg", with a broken Tcl/Tk install underneath it. Some third-party
imports (e.g. `prophet`) import `matplotlib.pyplot` as a side effect,
which resolves/locks in that default backend. Whichever test module
happens to trigger that first decides the backend for every other test
in the same pytest session -- e.g. `tests/test_experiment_scripts.py`
(which imports `experiments.scripts.run_baselines` -> `prophet`) sorts
alphabetically before `tests/test_figures.py`, so without this conftest
the figures tests would fail with a Tcl error depending on collection
order alone. Setting Agg here, in the root conftest.py (imported before
any test module), makes the outcome independent of that order.
"""
import matplotlib

matplotlib.use("Agg")
