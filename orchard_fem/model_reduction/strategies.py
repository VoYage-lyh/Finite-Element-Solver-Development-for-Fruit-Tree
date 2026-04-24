from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReducedBasis:
    vectors: list[list[float]] = field(default_factory=list)


class ReductionStrategy(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
