#include "orchard_solver/discretization/Assembler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "orchard_solver/discretization/BeamElement.h"
#include "orchard_solver/solver_core/StaticPreload.h"

namespace orchard {

namespace {

constexpr std::array<const char*, 6> kComponentLabels = {"ux", "uy", "uz", "rx", "ry", "rz"};
constexpr double kConstraintPenalty = 1.0e10;
constexpr double kGravityAcceleration = 9.81;

int translationalComponentIndex(const std::string& component) {
    if (component == "ux") {
        return 0;
    }
    if (component == "uy") {
        return 1;
    }
    if (component == "uz") {
        return 2;
    }

    throw std::runtime_error("Unsupported translational component: " + component);
}

int resolveNodeIndex(const std::vector<BranchNodeState>& nodes, const std::string& target_node) {
    if (nodes.empty()) {
        throw std::runtime_error("Branch has no discretized nodes");
    }

    if (target_node == "root") {
        return 0;
    }
    if (target_node == "tip") {
        return static_cast<int>(nodes.size()) - 1;
    }

    const int index = std::stoi(target_node);
    if (index < 0 || index >= static_cast<int>(nodes.size())) {
        throw std::runtime_error("Requested node index is out of range: " + target_node);
    }

    return index;
}

void scatterElementMatrix(
    DenseMatrix& global,
    const Matrix12& local,
    const std::array<int, 12>& dofs
) {
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t col = 0; col < 12; ++col) {
            global.add(
                static_cast<std::size_t>(dofs[row]),
                static_cast<std::size_t>(dofs[col]),
                local[row][col]
            );
        }
    }
}

void addPairPenalty(DenseMatrix& matrix, const int first_dof, const int second_dof, const double penalty) {
    matrix.add(first_dof, first_dof, penalty);
    matrix.add(first_dof, second_dof, -penalty);
    matrix.add(second_dof, first_dof, -penalty);
    matrix.add(second_dof, second_dof, penalty);
}

double computeFallbackDampingRatio(const OrchardModel& model) {
    double weighted_sum = 0.0;
    double weight = 0.0;

    for (const auto& branch : model.branches) {
        const auto properties = branch.reportAverageProperties(model.materials);
        const double branch_mass = properties.average_mass_per_length * std::max(properties.length, 1.0e-6);
        weighted_sum += branch_mass * properties.average_damping_ratio;
        weight += branch_mass;
    }

    return weight > 0.0 ? weighted_sum / weight : 0.0;
}

void applyRayleighDamping(const OrchardModel& model, DynamicSystem& system) {
    double alpha = model.analysis.rayleigh_alpha;
    double beta = model.analysis.rayleigh_beta;

    if (std::abs(alpha) < 1.0e-14 && std::abs(beta) < 1.0e-14) {
        const double zeta = computeFallbackDampingRatio(model);
        const double omega_ref = 2.0 * 3.14159265358979323846 * std::max(model.analysis.frequency_start_hz, 0.1);
        beta = omega_ref > 0.0 ? (2.0 * zeta / omega_ref) : 0.0;
    }

    for (std::size_t row = 0; row < system.mass.rows(); ++row) {
        for (std::size_t col = 0; col < system.mass.cols(); ++col) {
            system.damping.add(row, col, (alpha * system.mass(row, col)) + (beta * system.stiffness(row, col)));
        }
    }
}

} // namespace

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

const std::vector<BranchNodeState>& AssembledModel::requireBranchNodes(const std::string& branch_id) const {
    const auto it = branch_nodes.find(branch_id);
    if (it == branch_nodes.end()) {
        throw std::runtime_error("Unknown branch node set: " + branch_id);
    }

    return it->second;
}

int AssembledModel::requireBranchDof(const std::string& branch_id, const int node_index, const std::string& component) const {
    const auto& nodes = requireBranchNodes(branch_id);
    if (node_index < 0 || node_index >= static_cast<int>(nodes.size())) {
        throw std::runtime_error("Branch node index out of range for branch: " + branch_id);
    }

    const auto component_index = translationalComponentIndex(component);
    return nodes[static_cast<std::size_t>(node_index)].dofs[static_cast<std::size_t>(component_index)];
}

AssembledModel StructuralAssembler::assemble(const OrchardModel& model) const {
    DOFManager dof_manager;
    AssembledModel assembled;

    for (const auto& branch : model.branches) {
        const int num_elements = std::max(branch.discretizationHint().num_elements, 1);
        auto& nodes = assembled.branch_nodes[branch.id()];
        nodes.reserve(static_cast<std::size_t>(num_elements + 1));

        for (int node_index = 0; node_index <= num_elements; ++node_index) {
            const double station = static_cast<double>(node_index) / static_cast<double>(num_elements);
            BranchNodeState node;
            node.label_prefix = "branch:" + branch.id() + ":n" + std::to_string(node_index);
            node.position = branch.path().pointAt(station);
            node.station = station;

            for (std::size_t component = 0; component < kComponentLabels.size(); ++component) {
                node.dofs[component] = dof_manager.registerLabel(node.label_prefix + ":" + kComponentLabels[component]);
            }

            nodes.push_back(node);
        }
    }

    for (const auto& fruit : model.fruits) {
        assembled.fruit_dofs[fruit.id] = dof_manager.registerLabel("fruit:" + fruit.id);
    }

    const auto dof_count = dof_manager.size();
    assembled.system.mass = DenseMatrix(dof_count, dof_count);
    assembled.system.damping = DenseMatrix(dof_count, dof_count);
    assembled.system.stiffness = DenseMatrix(dof_count, dof_count);
    assembled.system.dof_labels = dof_manager.labels();
    assembled.system.gravity_load.assign(dof_count, 0.0);

    const bool apply_gravity_prestress = model.analysis.include_gravity_prestress;
    const Vec3 gravity_direction = apply_gravity_prestress
        ? normalize(model.analysis.gravity_direction)
        : Vec3 {0.0, 0.0, -1.0};

    for (const auto& branch : model.branches) {
        const auto& nodes = assembled.branch_nodes.at(branch.id());
        auto& elements = assembled.branch_elements[branch.id()];

        for (std::size_t element_index = 0; element_index + 1 < nodes.size(); ++element_index) {
            const auto& first = nodes[element_index];
            const auto& second = nodes[element_index + 1];

            const auto first_state = branch.evaluateSectionState(model.materials, first.station);
            const auto second_state = branch.evaluateSectionState(model.materials, second.station);

            const double area = 0.5 * (first_state.area + second_state.area);
            const double iy = 0.5 * (first_state.ix + second_state.ix);
            const double iz = 0.5 * (first_state.iy + second_state.iy);
            const double polar_moment = std::max(0.5 * (first_state.polar_moment + second_state.polar_moment), iy + iz);
            const double mass_per_length = 0.5 * (first_state.mass_per_length + second_state.mass_per_length);
            const double length = distance(first.position, second.position);

            BeamElementProperties properties;
            properties.youngs_modulus = 0.5 * (first_state.effective_youngs_modulus + second_state.effective_youngs_modulus);
            properties.shear_modulus = 0.5 * (first_state.effective_shear_modulus + second_state.effective_shear_modulus);
            properties.area = area;
            properties.iy = iy;
            properties.iz = iz;
            properties.torsion_constant = polar_moment; // TODO: replace Ix + Iy approximation with orchard-specific torsion estimate.
            properties.density = area > 0.0 ? mass_per_length / area : 0.0;
            properties.length = length;

            const auto local_stiffness = BeamElement::buildLocalStiffnessMatrix(properties);
            const auto local_mass = BeamElement::buildLocalMassMatrix(properties);
            const auto transformation = BeamElement::buildTransformationMatrix(first.position, second.position);
            const auto global_stiffness = BeamElement::transformToGlobal(local_stiffness, transformation);
            const auto global_mass = BeamElement::transformToGlobal(local_mass, transformation);

            std::array<int, 12> element_dofs {};
            for (std::size_t component = 0; component < 6; ++component) {
                element_dofs[component] = first.dofs[component];
                element_dofs[component + 6] = second.dofs[component];
            }

            scatterElementMatrix(assembled.system.stiffness, global_stiffness, element_dofs);
            scatterElementMatrix(assembled.system.mass, global_mass, element_dofs);
            elements.push_back(BranchElementState {
                branch.id(),
                static_cast<int>(element_index),
                element_dofs,
                transformation,
                length,
                properties.youngs_modulus * properties.area
            });

            if (apply_gravity_prestress) {
                const double nodal_scale = 0.5 * mass_per_length * kGravityAcceleration * length;
                const Vec3 nodal_force = gravity_direction * nodal_scale;
                for (const auto& node : {first, second}) {
                    assembled.system.gravity_load[static_cast<std::size_t>(node.dofs[0])] += nodal_force.x;
                    assembled.system.gravity_load[static_cast<std::size_t>(node.dofs[1])] += nodal_force.y;
                    assembled.system.gravity_load[static_cast<std::size_t>(node.dofs[2])] += nodal_force.z;
                }
            }
        }
    }

    for (const auto& branch : model.branches) {
        if (!branch.parentBranchId().has_value()) {
            continue;
        }

        const auto& child_nodes = assembled.requireBranchNodes(branch.id());
        const auto& parent_nodes = assembled.requireBranchNodes(*branch.parentBranchId());
        const auto& child_root = child_nodes.front();

        std::size_t nearest_parent_index = 0;
        double best_distance = distance(child_root.position, parent_nodes.front().position);
        for (std::size_t parent_index = 1; parent_index < parent_nodes.size(); ++parent_index) {
            const double candidate = distance(child_root.position, parent_nodes[parent_index].position);
            if (candidate < best_distance) {
                best_distance = candidate;
                nearest_parent_index = parent_index;
            }
        }

        double penalty = kConstraintPenalty;
        if (const auto* joint = model.findJointForChild(branch.id())) {
            penalty *= std::max(joint->linear_stiffness_scale, 1.0e-6);
        }

        for (std::size_t component = 0; component < 6; ++component) {
            addPairPenalty(
                assembled.system.stiffness,
                child_root.dofs[component],
                parent_nodes[nearest_parent_index].dofs[component],
                penalty
            );
        }
    }

    if (!model.analysis.auto_nonlinear_levels.empty()) {
        for (const auto& branch : model.branches) {
            if (!branch.parentBranchId().has_value()) {
                continue;
            }
            if (std::find(
                    model.analysis.auto_nonlinear_levels.begin(),
                    model.analysis.auto_nonlinear_levels.end(),
                    branch.level()
                ) == model.analysis.auto_nonlinear_levels.end()) {
                continue;
            }
            if (model.findJointForChild(branch.id()) != nullptr) {
                continue;
            }

            const auto& child_nodes = assembled.requireBranchNodes(branch.id());
            const auto& parent_nodes = assembled.requireBranchNodes(*branch.parentBranchId());
            const auto& child_root = child_nodes.front();

            std::size_t nearest_parent_index = 0;
            double best_distance = distance(child_root.position, parent_nodes.front().position);
            for (std::size_t parent_index = 1; parent_index < parent_nodes.size(); ++parent_index) {
                const double candidate = distance(child_root.position, parent_nodes[parent_index].position);
                if (candidate < best_distance) {
                    best_distance = candidate;
                    nearest_parent_index = parent_index;
                }
            }

            assembled.system.nonlinear_links.push_back(NonlinearLink {
                "auto_joint:" + branch.id(),
                child_root.dofs[0],
                parent_nodes[nearest_parent_index].dofs[0],
                NonlinearLinkKind::CubicSpring,
                0.0,
                model.analysis.auto_nonlinear_cubic_scale,
                0.0,
                0.0
            });
        }
    }

    for (const auto& clamp : model.clamps) {
        const auto& root_nodes = assembled.requireBranchNodes(clamp.branch_id);
        const auto& root = root_nodes.front();

        for (const auto dof : root.dofs) {
            assembled.system.stiffness.add(dof, dof, kConstraintPenalty);
        }

        if (std::abs(clamp.cubic_stiffness) > 0.0) {
            assembled.system.nonlinear_links.push_back(NonlinearLink {
                "clamp:" + clamp.branch_id,
                root.dofs[0],
                -1,
                NonlinearLinkKind::CubicSpring,
                kConstraintPenalty,
                clamp.cubic_stiffness,
                0.0,
                0.0
            });
        }
    }

    for (const auto& fruit : model.fruits) {
        const int fruit_dof = assembled.fruit_dofs.at(fruit.id);
        const auto& branch_nodes = assembled.requireBranchNodes(fruit.branch_id);

        std::size_t nearest_branch_node = 0;
        double best_station_distance = std::abs(branch_nodes.front().station - fruit.location_s);
        for (std::size_t node_index = 1; node_index < branch_nodes.size(); ++node_index) {
            const double candidate = std::abs(branch_nodes[node_index].station - fruit.location_s);
            if (candidate < best_station_distance) {
                best_station_distance = candidate;
                nearest_branch_node = node_index;
            }
        }

        const int branch_dof = branch_nodes[nearest_branch_node].dofs[0];
        assembled.system.mass.add(fruit_dof, fruit_dof, std::max(fruit.mass, 1.0e-9));

        const double stiffness = std::max(fruit.stiffness, 1.0e-6);
        assembled.system.stiffness.add(fruit_dof, fruit_dof, stiffness);
        assembled.system.stiffness.add(fruit_dof, branch_dof, -stiffness);
        assembled.system.stiffness.add(branch_dof, fruit_dof, -stiffness);
        assembled.system.stiffness.add(branch_dof, branch_dof, stiffness);

        const double damping = std::max(fruit.damping, 0.0);
        assembled.system.damping.add(fruit_dof, fruit_dof, damping);
        assembled.system.damping.add(fruit_dof, branch_dof, -damping);
        assembled.system.damping.add(branch_dof, fruit_dof, -damping);
        assembled.system.damping.add(branch_dof, branch_dof, damping);
    }

    if (apply_gravity_prestress) {
        const auto axial_forces = computeGravityAxialForces(assembled, assembled.system.gravity_load);
        for (const auto& branch : model.branches) {
            const auto& elements = assembled.branch_elements.at(branch.id());
            const auto& forces = axial_forces.at(branch.id());
            for (std::size_t index = 0; index < elements.size(); ++index) {
                const auto local_geometric = BeamElement::buildLocalGeometricStiffnessMatrix(
                    forces[index],
                    elements[index].length
                );
                const auto global_geometric = BeamElement::transformToGlobal(
                    local_geometric,
                    elements[index].transformation
                );
                scatterElementMatrix(assembled.system.stiffness, global_geometric, elements[index].dofs);
            }
        }
    }

    applyRayleighDamping(model, assembled.system);

    assembled.excitation_dof = assembled.requireBranchDof(
        model.excitation.target_branch_id,
        resolveNodeIndex(assembled.requireBranchNodes(model.excitation.target_branch_id), model.excitation.target_node),
        model.excitation.target_component
    );

    for (const auto& observation : model.observations) {
        if (observation.target_type == "branch") {
            const auto& nodes = assembled.requireBranchNodes(observation.target_id);
            const int node_index = resolveNodeIndex(nodes, observation.target_node);
            if (observation.target_components.size() <= 1U) {
                assembled.observation_names.push_back(observation.id);
                assembled.observation_dofs.push_back(
                    assembled.requireBranchDof(
                        observation.target_id,
                        node_index,
                        observation.target_components.front()
                    )
                );
            } else {
                for (const auto& component : observation.target_components) {
                    assembled.observation_names.push_back(observation.id + "_" + component);
                    assembled.observation_dofs.push_back(
                        assembled.requireBranchDof(
                            observation.target_id,
                            node_index,
                            component
                        )
                    );
                }
            }
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
