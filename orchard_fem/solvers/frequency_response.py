from __future__ import annotations

from dataclasses import dataclass


def petsc_available() -> bool:
    try:
        import petsc4py  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class FrequencyResponseRequest:
    model_path: str
    output_csv: str


class PETScFrequencyResponseSolver:
    def solve(self, request: FrequencyResponseRequest) -> None:
        if not petsc_available():
            raise RuntimeError(
                "PETSc backend is not available. Install the conda environment from "
                "config/fenicsx_pinn_environment.yml before enabling the Python frequency-response path."
            )

        raise NotImplementedError(
            "P1 frequency-response port is not implemented yet. The C++ orchard_cli path remains active."
        )
