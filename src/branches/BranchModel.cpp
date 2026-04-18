#include "orchard_solver/branches/BranchModel.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace orchard {

namespace {

struct SectionConstitutiveMetrics {
    double area {0.0};
    double ix {0.0};
    double iy {0.0};
    double mass_per_length {0.0};
    double flexural_rigidity {0.0};
    double damping_ratio {0.0};
};

SectionConstitutiveMetrics evaluateSection(
    const SectionProperties& properties,
    const MaterialLibrary& materials
) {
    SectionConstitutiveMetrics metrics {};
    metrics.area = properties.total_area;
    metrics.ix = properties.ix_centroid;
    metrics.iy = properties.iy_centroid;

    double damping_weight = 0.0;

    for (const auto& region : properties.regions) {
        const auto& material = materials.require(region.material_id);
        const double youngs_modulus = material.tangentModulus(0.0);
        const double dx = region.centroid.x - properties.centroid.x;
        const double dy = region.centroid.y - properties.centroid.y;
        const double ix_about_section = region.ix_centroid + (region.area * dy * dy);
        const double iy_about_section = region.iy_centroid + (region.area * dx * dx);
        const double i_equivalent = 0.5 * (ix_about_section + iy_about_section);

        metrics.mass_per_length += material.properties().density * region.area;
        metrics.flexural_rigidity += youngs_modulus * i_equivalent;
        damping_weight += material.properties().damping_ratio * material.properties().density * region.area;
    }

    if (metrics.mass_per_length > 0.0) {
        metrics.damping_ratio = damping_weight / metrics.mass_per_length;
    }

    return metrics;
}

double trapezoidalAverage(const std::vector<double>& stations, const std::vector<double>& values) {
    if (stations.size() != values.size() || stations.empty()) {
        throw std::runtime_error("Station/value arrays must be aligned and non-empty");
    }

    if (stations.size() == 1U) {
        return values.front();
    }

    double weighted_sum = 0.0;
    double span = stations.back() - stations.front();
    if (span <= 0.0) {
        span = 1.0;
    }

    for (std::size_t i = 0; i + 1 < stations.size(); ++i) {
        const double ds = stations[i + 1] - stations[i];
        weighted_sum += 0.5 * (values[i] + values[i + 1]) * ds;
    }

    return weighted_sum / span;
}

} // namespace

BranchComponent::BranchComponent(
    std::string id,
    std::optional<std::string> parent_branch_id,
    const int level,
    BranchPath path,
    MeasuredSectionSeries section_series,
    BranchDiscretizationHint discretization_hint
)
    : id_(std::move(id)),
      parent_branch_id_(std::move(parent_branch_id)),
      level_(level),
      path_(std::move(path)),
      section_series_(std::move(section_series)),
      discretization_hint_(discretization_hint) {
}

const std::string& BranchComponent::id() const noexcept {
    return id_;
}

const std::optional<std::string>& BranchComponent::parentBranchId() const noexcept {
    return parent_branch_id_;
}

int BranchComponent::level() const noexcept {
    return level_;
}

const BranchPath& BranchComponent::path() const noexcept {
    return path_;
}

const MeasuredSectionSeries& BranchComponent::sectionSeries() const noexcept {
    return section_series_;
}

const BranchDiscretizationHint& BranchComponent::discretizationHint() const noexcept {
    return discretization_hint_;
}

BranchEffectiveProperties BranchComponent::computeEffectiveProperties(const MaterialLibrary& materials) const {
    const auto& profiles = section_series_.profiles();
    if (profiles.empty()) {
        throw std::runtime_error("Branch '" + id_ + "' does not define any section stations");
    }

    std::vector<double> stations;
    std::vector<double> areas;
    std::vector<double> ixs;
    std::vector<double> iys;
    std::vector<double> mass_per_length_values;
    std::vector<double> flexural_rigidity_values;
    std::vector<double> damping_ratio_values;

    stations.reserve(profiles.size());
    areas.reserve(profiles.size());
    ixs.reserve(profiles.size());
    iys.reserve(profiles.size());
    mass_per_length_values.reserve(profiles.size());
    flexural_rigidity_values.reserve(profiles.size());
    damping_ratio_values.reserve(profiles.size());

    for (const auto& profile : profiles) {
        const auto geometry = profile->evaluate();
        const auto metrics = evaluateSection(geometry, materials);
        stations.push_back(profile->station());
        areas.push_back(metrics.area);
        ixs.push_back(metrics.ix);
        iys.push_back(metrics.iy);
        mass_per_length_values.push_back(metrics.mass_per_length);
        flexural_rigidity_values.push_back(metrics.flexural_rigidity);
        damping_ratio_values.push_back(metrics.damping_ratio);
    }

    BranchEffectiveProperties result {};
    result.length = path_.length();
    result.average_area = trapezoidalAverage(stations, areas);
    result.average_ix = trapezoidalAverage(stations, ixs);
    result.average_iy = trapezoidalAverage(stations, iys);
    result.average_mass_per_length = trapezoidalAverage(stations, mass_per_length_values);
    result.average_flexural_rigidity = trapezoidalAverage(stations, flexural_rigidity_values);

    const double average_damping_ratio = trapezoidalAverage(stations, damping_ratio_values);
    const double safe_length = std::max(result.length, 1.0e-6);
    const double refinement_boost = 1.0 + (0.10 * static_cast<double>(std::max(discretization_hint_.refinement_level - 1, 0)));
    const double hotspot_boost = discretization_hint_.hotspot ? 1.10 : 1.0;

    result.equivalent_mass = result.average_mass_per_length * safe_length;
    result.equivalent_stiffness = (3.0 * result.average_flexural_rigidity / std::pow(safe_length, 3)) * refinement_boost * hotspot_boost;
    result.equivalent_damping = 2.0 * average_damping_ratio * std::sqrt(std::max(result.equivalent_stiffness * result.equivalent_mass, 0.0));

    return result;
}

} // namespace orchard
