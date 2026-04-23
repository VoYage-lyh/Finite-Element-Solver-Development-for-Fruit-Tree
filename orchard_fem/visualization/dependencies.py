from __future__ import annotations

import os
import tempfile
from pathlib import Path

plt = None
np = None

PLOT_INSTALL_HINT = (
    "Visualization requires numpy and matplotlib. Install the repository test extras with "
    '`python -m pip install -e ".[ubuntu-test]"` or create the conda environment from '
    "`config/fenicsx_pinn_environment.yml`."
)


class MissingDependencyError(RuntimeError):
    pass


def require_plotting_dependencies(show: bool):
    global plt, np

    if plt is not None and np is not None:
        return plt, np

    cache_root = Path(tempfile.gettempdir()) / "orchard-mpl-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as _plt
        import numpy as _np
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"{PLOT_INSTALL_HINT} Missing module: {exc.name}."
        ) from exc

    plt = _plt
    np = _np
    return plt, np
