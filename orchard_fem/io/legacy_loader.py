from __future__ import annotations

from orchard_fem.io.json_schema import build_topology_from_legacy_model, load_legacy_model
from orchard_fem.model import (
    AnalysisMode,
    AnalysisSettings,
    BranchDefinition,
    BranchDiscretizationHint,
    ClampBoundaryCondition,
    ExcitationKind,
    HarmonicExcitation,
    JointDefinition,
    JointLawDefinition,
    JointLawKind,
    MaterialModelKind,
    MaterialProperties,
    OrchardMetadata,
    OrchardModel,
    FruitAttachment,
    parse_section_series,
    parse_vec3,
)
from orchard_fem.topology.tree import BranchPath, ObservationPoint
from orchard_fem.cross_section.tissue import TissueType


def _parse_observation_target_components(observation: dict) -> list[str]:
    if "target_components" in observation:
        value = observation["target_components"]
        if not isinstance(value, list):
            raise ValueError("observations[].target_components must be a list of strings")
        components = [str(component) for component in value]
        if not components:
            raise ValueError("observations[].target_components must not be empty")
        return components

    if "target_component" in observation:
        return [str(observation["target_component"])]

    return ["ux"]


def load_orchard_model(file_path: str) -> OrchardModel:
    payload = load_legacy_model(file_path)
    topology = build_topology_from_legacy_model(payload)

    metadata_payload = payload.get("metadata", {})
    metadata = OrchardMetadata(
        name=str(metadata_payload.get("name", "")),
        cultivar=str(metadata_payload.get("cultivar", "")),
    )

    materials = [
        MaterialProperties(
            material_id=str(material["id"]),
            tissue=TissueType(material["tissue"]),
            model=MaterialModelKind(material.get("model", "linear")),
            density=float(material["density"]),
            youngs_modulus=float(material["youngs_modulus"]),
            poisson_ratio=float(material.get("poisson_ratio", 0.3)),
            damping_ratio=float(material["damping_ratio"]),
            nonlinear_alpha=float(material.get("nonlinear_alpha", 0.0)),
        )
        for material in payload.get("materials", [])
    ]
    available_material_ids = {material.material_id for material in materials}

    branches = [
        BranchDefinition(
            branch_id=str(branch["id"]),
            parent_branch_id=branch.get("parent_branch_id"),
            level=int(branch["level"]),
            path=BranchPath(
                start=parse_vec3(branch["start"]),
                end=parse_vec3(branch["end"]),
            ),
            section_series=parse_section_series(
                branch["stations"],
                available_material_ids=available_material_ids,
            ),
            discretization=BranchDiscretizationHint(
                num_elements=int(branch.get("discretization", {}).get("num_elements", 4)),
                hotspot=bool(branch.get("discretization", {}).get("hotspot", False)),
            ),
        )
        for branch in payload.get("branches", [])
    ]

    joints = []
    for joint in payload.get("joints", []):
        law_payload = joint.get("law", {})
        law_kind = JointLawKind.NONE if not law_payload else JointLawKind(law_payload["type"])
        joints.append(
            JointDefinition(
                joint_id=str(joint["id"]),
                parent_branch_id=str(joint["parent_branch_id"]),
                child_branch_id=str(joint["child_branch_id"]),
                linear_stiffness_scale=float(joint.get("linear_stiffness_scale", 1.0)),
                law=JointLawDefinition(
                    kind=law_kind,
                    linear_scale=float(law_payload.get("linear_scale", 1.0)),
                    cubic_scale=float(law_payload.get("cubic_scale", 0.0)),
                    open_scale=float(law_payload.get("open_scale", 1.0)),
                    gap_threshold=float(law_payload.get("gap_threshold", 0.0)),
                ),
            )
        )

    fruits = [
        FruitAttachment(
            fruit_id=str(fruit["id"]),
            branch_id=str(fruit["branch_id"]),
            location_s=float(fruit["location_s"]),
            mass=float(fruit["mass"]),
            stiffness=float(fruit["stiffness"]),
            damping=float(fruit["damping"]),
        )
        for fruit in payload.get("fruits", [])
    ]

    clamps = [
        ClampBoundaryCondition(
            branch_id=str(clamp["branch_id"]),
            support_stiffness=float(clamp["support_stiffness"]),
            support_damping=float(clamp["support_damping"]),
            cubic_stiffness=float(clamp.get("cubic_stiffness", 0.0)),
        )
        for clamp in payload.get("clamps", [])
    ]

    excitation_payload = payload["excitation"]
    excitation = HarmonicExcitation(
        kind=ExcitationKind(excitation_payload["kind"]),
        target_branch_id=str(excitation_payload["target_branch_id"]),
        target_node=str(excitation_payload.get("target_node", "tip")),
        target_component=str(excitation_payload.get("target_component", "ux")),
        amplitude=float(excitation_payload["amplitude"]),
        phase_degrees=float(excitation_payload["phase_degrees"]),
        driving_frequency_hz=float(excitation_payload.get("driving_frequency_hz", 0.0)),
    )

    analysis_payload = payload["analysis"]
    gravity_direction = parse_vec3(analysis_payload.get("gravity_direction", [0.0, 0.0, -1.0]))
    analysis = AnalysisSettings(
        mode=AnalysisMode(analysis_payload.get("mode", "frequency_response")),
        frequency_start_hz=float(analysis_payload["frequency_start_hz"]),
        frequency_end_hz=float(analysis_payload["frequency_end_hz"]),
        frequency_steps=int(analysis_payload["frequency_steps"]),
        time_step_seconds=float(analysis_payload.get("time_step_seconds", 0.002)),
        total_time_seconds=float(analysis_payload.get("total_time_seconds", 1.0)),
        output_stride=int(analysis_payload.get("output_stride", 1)),
        max_nonlinear_iterations=int(analysis_payload.get("max_nonlinear_iterations", 12)),
        nonlinear_tolerance=float(analysis_payload.get("nonlinear_tolerance", 1.0e-8)),
        rayleigh_alpha=float(analysis_payload.get("rayleigh_alpha", 0.0)),
        rayleigh_beta=float(analysis_payload.get("rayleigh_beta", 1.0e-4)),
        auto_nonlinear_levels=[int(level) for level in analysis_payload.get("auto_nonlinear_levels", [])],
        auto_nonlinear_cubic_scale=float(analysis_payload.get("auto_nonlinear_cubic_scale", 0.0)),
        include_gravity_prestress=bool(analysis_payload.get("include_gravity_prestress", False)),
        gravity_direction=(gravity_direction.x, gravity_direction.y, gravity_direction.z),
        output_csv=str(analysis_payload.get("output_csv", "frequency_response.csv")),
    )

    observations = [
        ObservationPoint(
            observation_id=str(observation["id"]),
            target_type=str(observation["target_type"]),
            target_id=str(observation["target_id"]),
            target_node=str(observation.get("target_node", "tip")),
            target_components=_parse_observation_target_components(observation),
        )
        for observation in payload.get("observations", [])
    ]

    return OrchardModel(
        metadata=metadata,
        materials=materials,
        topology=topology,
        branches=branches,
        joints=joints,
        fruits=fruits,
        clamps=clamps,
        excitation=excitation,
        analysis=analysis,
        observations=observations,
    )
