#include "orchard_solver/discretization/Assembler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace orchard {

int DOFManager::registerLabel(const std::string& label) {
    const auto it = label_to_index_.find(label);
    if (it != label_to_index_.end()) {
        return it->second;
    }

    const int index = static_cast<int>(labels_.size());
    labels_.push_back(label);
    label_to_index_[label] = index;
    return index;
}

int DOFManager::require(const std::string& label) const {
    const auto it = label_to_index_.find(label);
    if (it == label_to_index_.end()) {
        throw std::runtime_error("Unknown DOF label: " + label);
    }

    return it->second;
}

std::size_t DOFManager::size() const noexcept {
    return labels_.size();
}

const std::vector<std::string>& DOFManager::labels() const noexcept {
    return labels_;
}

AssembledModel StructuralAssembler::assemble(const OrchardModel& model) const {
    DOFManager dof_manager;
    AssembledModel assembled;

    for (const auto& branch : model.branches) {
        assembled.branch_dofs[branch.id()] = dof_manager.registerLabel("branch:" + branch.id());
    }

    for (const auto& fruit : model.fruits) {
        assembled.fruit_dofs[fruit.id] = dof_manager.registerLabel("fruit:" + fruit.id);
    }

    const auto dof_count = dof_manager.size();
    assembled.system.mass = DenseMatrix(dof_count, dof_count);
    assembled.system.damping = DenseMatrix(dof_count, dof_count);
    assembled.system.stiffness = DenseMatrix(dof_count, dof_count);
    assembled.system.dof_labels = dof_manager.labels();

    std::unordered_map<std::string, BranchEffectiveProperties> branch_effective_properties;
    for (const auto& branch : model.branches) {
        branch_effective_properties[branch.id()] = branch.computeEffectiveProperties(model.materials);
    }

    for (const auto& branch : model.branches) {
        const int dof = assembled.branch_dofs.at(branch.id());
        const auto& effective = branch_effective_properties.at(branch.id());
        assembled.system.mass.add(dof, dof, std::max(effective.equivalent_mass, 1.0e-9));

        const double base_branch_stiffness = std::max(effective.equivalent_stiffness, 1.0);
        double branch_stiffness = base_branch_stiffness;
        double branch_damping = std::max(effective.equivalent_damping, 1.0e-6);

        if (branch.parentBranchId().has_value()) {
            const int parent_dof = assembled.branch_dofs.at(*branch.parentBranchId());
            if (const auto* joint = model.findJointForChild(branch.id())) {
                const auto parameters = joint->parameters();
                const double scale = std::max(joint->linearizedScale(), 0.05);
                branch_stiffness *= scale;
                branch_damping *= scale;

                if (parameters.kind == JointLawKind::Cubic && std::abs(parameters.cubic_scale) > 0.0) {
                    assembled.system.nonlinear_links.push_back(NonlinearLink {
                        "joint:" + joint->id,
                        dof,
                        parent_dof,
                        NonlinearLinkKind::CubicSpring,
                        branch_stiffness,
                        base_branch_stiffness * joint->linear_stiffness_scale * parameters.cubic_scale,
                        0.0,
                        0.0
                    });
                } else if (parameters.kind == JointLawKind::Gap) {
                    assembled.system.nonlinear_links.push_back(NonlinearLink {
                        "joint:" + joint->id,
                        dof,
                        parent_dof,
                        NonlinearLinkKind::GapSpring,
                        branch_stiffness,
                        0.0,
                        base_branch_stiffness * joint->linear_stiffness_scale * parameters.open_scale,
                        parameters.gap_threshold
                    });
                }
            }

            assembled.system.stiffness.add(dof, dof, branch_stiffness);
            assembled.system.stiffness.add(dof, parent_dof, -branch_stiffness);
            assembled.system.stiffness.add(parent_dof, dof, -branch_stiffness);
            assembled.system.stiffness.add(parent_dof, parent_dof, branch_stiffness);

            assembled.system.damping.add(dof, dof, branch_damping);
            assembled.system.damping.add(dof, parent_dof, -branch_damping);
            assembled.system.damping.add(parent_dof, dof, -branch_damping);
            assembled.system.damping.add(parent_dof, parent_dof, branch_damping);
        } else {
            const auto* clamp = model.findClamp(branch.id());
            const double clamp_stiffness = clamp ? clamp->support_stiffness : 0.0;
            const double clamp_damping = clamp ? clamp->support_damping : 0.0;

            assembled.system.stiffness.add(dof, dof, branch_stiffness + clamp_stiffness);
            assembled.system.damping.add(dof, dof, branch_damping + clamp_damping);

            if (clamp && std::abs(clamp->cubic_stiffness) > 0.0) {
                assembled.system.nonlinear_links.push_back(NonlinearLink {
                    "clamp:" + branch.id(),
                    dof,
                    -1,
                    NonlinearLinkKind::CubicSpring,
                    clamp_stiffness,
                    clamp->cubic_stiffness,
                    0.0,
                    0.0
                });
            }
        }
    }

    for (const auto& fruit : model.fruits) {
        const int fruit_dof = assembled.fruit_dofs.at(fruit.id);
        const int branch_dof = assembled.branch_dofs.at(fruit.branch_id);

        assembled.system.mass.add(fruit_dof, fruit_dof, std::max(fruit.mass, 1.0e-9));

        const double stiffness = std::max(fruit.stiffness, 1.0e-6);
        const double damping = std::max(fruit.damping, 1.0e-8);

        assembled.system.stiffness.add(fruit_dof, fruit_dof, stiffness);
        assembled.system.stiffness.add(fruit_dof, branch_dof, -stiffness);
        assembled.system.stiffness.add(branch_dof, fruit_dof, -stiffness);
        assembled.system.stiffness.add(branch_dof, branch_dof, stiffness);

        assembled.system.damping.add(fruit_dof, fruit_dof, damping);
        assembled.system.damping.add(fruit_dof, branch_dof, -damping);
        assembled.system.damping.add(branch_dof, fruit_dof, -damping);
        assembled.system.damping.add(branch_dof, branch_dof, damping);
    }

    assembled.excitation_dof = assembled.branch_dofs.at(model.excitation.target_branch_id);

    for (const auto& observation : model.observations) {
        if (observation.target_type == "branch") {
            assembled.observation_names.push_back(observation.id);
            assembled.observation_dofs.push_back(assembled.branch_dofs.at(observation.target_id));
        } else if (observation.target_type == "fruit") {
            assembled.observation_names.push_back(observation.id);
            assembled.observation_dofs.push_back(assembled.fruit_dofs.at(observation.target_id));
        } else {
            throw std::runtime_error("Unsupported observation target type: " + observation.target_type);
        }
    }

    if (assembled.observation_dofs.empty()) {
        assembled.observation_names.push_back("excitation_branch");
        assembled.observation_dofs.push_back(assembled.excitation_dof);
    }

    return assembled;
}

} // namespace orchard
