#pragma once

#include <optional>
#include <string>

#include "orchard_solver/cross_section/CrossSection.h"
#include "orchard_solver/geometry_topology/TreeTopology.h"
#include "orchard_solver/materials/Materials.h"

namespace orchard {

struct BranchDiscretizationHint {
    int num_elements {4};
    bool hotspot {false};
};

struct BranchAverageProperties {
    double length {0.0};
    double average_area {0.0};
    double average_ix {0.0};
    double average_iy {0.0};
    double average_polar_moment {0.0};
    double average_mass_per_length {0.0};
    double average_youngs_modulus {0.0};
    double average_shear_modulus {0.0};
    double average_damping_ratio {0.0};
};

struct BranchSectionState {
    double station {0.0};
    double area {0.0};
    double ix {0.0};
    double iy {0.0};
    double polar_moment {0.0};
    double mass_per_length {0.0};
    double effective_youngs_modulus {0.0};
    double effective_shear_modulus {0.0};
    double effective_poisson_ratio {0.3};
    double damping_ratio {0.0};
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

    [[nodiscard]] BranchAverageProperties reportAverageProperties(const MaterialLibrary& materials) const;
    [[nodiscard]] BranchSectionState evaluateSectionState(const MaterialLibrary& materials, double station) const;

private:
    std::string id_;
    std::optional<std::string> parent_branch_id_;
    int level_ {0};
    BranchPath path_;
    MeasuredSectionSeries section_series_;
    BranchDiscretizationHint discretization_hint_ {};
};

} // namespace orchard
