#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

namespace orchard {

struct BranchNodeState {
    std::string label_prefix;
    Vec3 position {};
    double station {0.0};
    std::array<int, 6> dofs {-1, -1, -1, -1, -1, -1};
};

class DOFManager {
public:
    int registerLabel(const std::string& label);
    [[nodiscard]] int require(const std::string& label) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept;

private:
    std::unordered_map<std::string, int> label_to_index_;
    std::vector<std::string> labels_;
};

struct AssembledModel {
    DynamicSystem system;
    std::unordered_map<std::string, std::vector<BranchNodeState>> branch_nodes;
    std::unordered_map<std::string, int> fruit_dofs;
    int excitation_dof {-1};
    std::vector<std::string> observation_names;
    std::vector<int> observation_dofs;

    [[nodiscard]] const std::vector<BranchNodeState>& requireBranchNodes(const std::string& branch_id) const;
    [[nodiscard]] int requireBranchDof(const std::string& branch_id, int node_index, const std::string& component) const;
};

class StructuralAssembler {
public:
    [[nodiscard]] AssembledModel assemble(const OrchardModel& model) const;
};

} // namespace orchard
