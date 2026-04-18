#include "orchard_solver/OrchardModel.h"

#include <stdexcept>

namespace orchard {

const BranchComponent& OrchardModel::requireBranch(const std::string& branch_id) const {
    for (const auto& branch : branches) {
        if (branch.id() == branch_id) {
            return branch;
        }
    }

    throw std::runtime_error("Unknown branch id: " + branch_id);
}

const JointComponent* OrchardModel::findJointForChild(const std::string& child_branch_id) const noexcept {
    for (const auto& joint : joints) {
        if (joint.child_branch_id == child_branch_id) {
            return &joint;
        }
    }

    return nullptr;
}

const ClampBoundaryCondition* OrchardModel::findClamp(const std::string& branch_id) const noexcept {
    for (const auto& clamp : clamps) {
        if (clamp.branch_id == branch_id) {
            return &clamp;
        }
    }

    return nullptr;
}

std::optional<ObservationPoint> OrchardModel::findObservation(const std::string& observation_id) const {
    for (const auto& observation : observations) {
        if (observation.id == observation_id) {
            return observation;
        }
    }

    return std::nullopt;
}

} // namespace orchard
