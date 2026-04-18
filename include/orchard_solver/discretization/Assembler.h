#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

namespace orchard {

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
    std::unordered_map<std::string, int> branch_dofs;
    std::unordered_map<std::string, int> fruit_dofs;
    int excitation_dof {-1};
    std::vector<std::string> observation_names;
    std::vector<int> observation_dofs;
};

class StructuralAssembler {
public:
    [[nodiscard]] AssembledModel assemble(const OrchardModel& model) const;
};

} // namespace orchard
