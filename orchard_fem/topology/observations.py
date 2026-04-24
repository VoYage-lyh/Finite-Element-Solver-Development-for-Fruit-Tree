from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationPoint:
    observation_id: str
    target_type: str
    target_id: str
    target_node: str = "tip"
    target_components: list[str] = field(default_factory=lambda: ["ux"])

    def __post_init__(self) -> None:
        normalized = [str(component) for component in self.target_components] or ["ux"]
        object.__setattr__(self, "target_components", normalized)

    @property
    def target_component(self) -> str:
        return self.target_components[0]
