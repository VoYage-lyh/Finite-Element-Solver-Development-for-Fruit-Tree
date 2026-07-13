from __future__ import annotations

import importlib.util

FENICSX_REQUIRED_MODULES = (
    "dolfinx",
    "basix",
    "ufl",
    "mpi4py",
)


def dolfinx_available() -> bool:
    return importlib.util.find_spec("dolfinx") is not None


def missing_fenicsx_modules() -> tuple[str, ...]:
    return tuple(
        module_name
        for module_name in FENICSX_REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
    )


def fenicsx_stack_available() -> bool:
    return not missing_fenicsx_modules()


def require_dolfinx() -> None:
    missing = missing_fenicsx_modules()
    if not missing:
        return
    raise RuntimeError(
        "FEniCSx backend is not available. Missing modules: "
        + ", ".join(missing)
        + ". Install the full FEniCSx stack by creating the "
        "`orchard-fenicsx` environment from `config/orchard_fenicsx.yml`."
    )
