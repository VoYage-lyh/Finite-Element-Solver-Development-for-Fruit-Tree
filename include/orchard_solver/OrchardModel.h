#pragma once

#include <optional>
#include <string>
#include <vector>

#include "orchard_solver/branches/BranchModel.h"
#include "orchard_solver/excitation_and_bc/Excitation.h"
#include "orchard_solver/fruits/Fruit.h"
#include "orchard_solver/geometry_topology/TreeTopology.h"
#include "orchard_solver/joints_and_bifurcations/Joints.h"
#include "orchard_solver/materials/Materials.h"

namespace orchard {

struct OrchardMetadata {
    std::string name;
    std::string cultivar;
};

class OrchardModel {
public:
    OrchardMetadata metadata;
    MaterialLibrary materials;
    TreeTopology topology;
    std::vector<BranchComponent> branches;
    std::vector<JointComponent> joints;
    std::vector<FruitAttachment> fruits;
    std::vector<ClampBoundaryCondition> clamps;
    HarmonicExcitation excitation;
    AnalysisSettings analysis;
    std::vector<ObservationPoint> observations;

    [[nodiscard]] const BranchComponent& requireBranch(const std::string& branch_id) const;
    [[nodiscard]] const JointComponent* findJointForChild(const std::string& child_branch_id) const noexcept;
    [[nodiscard]] const ClampBoundaryCondition* findClamp(const std::string& branch_id) const noexcept;
    [[nodiscard]] std::optional<ObservationPoint> findObservation(const std::string& observation_id) const;
};

} // namespace orchard
