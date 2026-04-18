#pragma once

#include <optional>
#include <string>

#include "orchard_solver/cross_section/CrossSection.h"
#include "orchard_solver/geometry_topology/TreeTopology.h"
#include "orchard_solver/materials/Materials.h"

namespace orchard {

struct BranchDiscretizationHint {
    int refinement_level {1};
    bool hotspot {false};
};

struct BranchEffectiveProperties {
    double length {0.0};
    double average_area {0.0};
    double average_ix {0.0};
    double average_iy {0.0};
    double average_mass_per_length {0.0};
    double average_flexural_rigidity {0.0};
    double equivalent_mass {0.0};
    double equivalent_stiffness {0.0};
    double equivalent_damping {0.0};
};

class BranchComponent {
public:
    BranchComponent(
        std::string id,
        std::optional<std::string> parent_branch_id,
        int level,
        BranchPath path,
        MeasuredSectionSeries section_series,
        BranchDiscretizationHint discretization_hint
    );

    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::optional<std::string>& parentBranchId() const noexcept;
    [[nodiscard]] int level() const noexcept;
    [[nodiscard]] const BranchPath& path() const noexcept;
    [[nodiscard]] const MeasuredSectionSeries& sectionSeries() const noexcept;
    [[nodiscard]] const BranchDiscretizationHint& discretizationHint() const noexcept;

    [[nodiscard]] BranchEffectiveProperties computeEffectiveProperties(const MaterialLibrary& materials) const;

private:
    std::string id_;
    std::optional<std::string> parent_branch_id_;
    int level_ {0};
    BranchPath path_;
    MeasuredSectionSeries section_series_;
    BranchDiscretizationHint discretization_hint_ {};
};

} // namespace orchard
