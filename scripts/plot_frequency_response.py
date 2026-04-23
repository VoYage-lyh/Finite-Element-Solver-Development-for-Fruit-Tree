from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchard_fem.postprocess.frequency_response import MissingDependencyError, main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingDependencyError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
