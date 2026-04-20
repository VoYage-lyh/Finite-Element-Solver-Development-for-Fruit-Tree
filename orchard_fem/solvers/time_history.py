from __future__ import annotations

from dataclasses import dataclass


def petsc_available() -> bool:
    try:
        import petsc4py  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class TimeHistoryRequest:
    model_path: str
    output_csv: str


class PETScTimeHistorySolver:
    def solve(self, request: TimeHistoryRequest) -> None:
        if not petsc_available():
            raise RuntimeError(
                "PETSc backend is not available. Install the conda environment from "
                "config/fenicsx_pinn_environment.yml before enabling the Python time-history path."
            )

        raise NotImplementedError(
            "P1 time-history port is not implemented yet. The C++ orchard_cli path remains active."
        )
