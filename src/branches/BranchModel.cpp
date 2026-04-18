#include "orchard_solver/branches/BranchModel.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace orchard {

namespace {

BranchSectionState evaluateProfileState(
    const double station,
    const SectionProperties& properties,
    const MaterialLibrary& materials
) {
    BranchSectionState state {};
    state.station = station;
    state.area = properties.total_area;
    state.ix = properties.ix_centroid;
    state.iy = properties.iy_centroid;
    state.polar_moment = properties.ix_centroid + properties.iy_centroid;

    double density_area_sum = 0.0;
    double modulus_area_sum = 0.0;
    double poisson_area_sum = 0.0;
    double damping_weight = 0.0;

    for (const auto& region : properties.regions) {
        const auto& material = materials.require(region.material_id);
        state.mass_per_length += material.properties().density * region.area;
        modulus_area_sum += material.tangentModulus(0.0) * region.area;
        poisson_area_sum += material.properties().poisson_ratio * region.area;
        damping_weight += material.properties().density * region.area * material.properties().damping_ratio;
        density_area_sum += region.area;
    }

    if (density_area_sum <= 0.0 || state.area <= 0.0) {
        throw std::runtime_error("Branch section state must have positive area");
    }

    state.effective_youngs_modulus = modulus_area_sum / density_area_sum;
    state.effective_poisson_ratio = poisson_area_sum / density_area_sum;
    state.effective_shear_modulus = state.effective_youngs_modulus / (2.0 * (1.0 + state.effective_poisson_ratio));
    state.damping_ratio = damping_weight / state.mass_per_length;

    return state;
}

std::vector<BranchSectionState> evaluateProfileStates(
    const MeasuredSectionSeries& section_series,
    const MaterialLibrary& materials
) {
    std::vector<BranchSectionState> states;
    states.reserve(section_series.profiles().size());

    for (const auto& profile : section_series.profiles()) {
        states.push_back(evaluateProfileState(profile->station(), profile->evaluate(), materials));
    }

    return states;
}

template <typename Getter>
double interpolateStateValue(
    const std::vector<BranchSectionState>& states,
    const double station,
    Getter getter
) {
    if (states.empty()) {
        throw std::runtime_error("Branch requires at least one measured section station");
    }

    if (states.size() == 1U || station <= states.front().station) {
        return getter(states.front());
    }

    if (station >= states.back().station) {
        return getter(states.back());
    }

    for (std::size_t i = 0; i + 1 < states.size(); ++i) {
        const auto& left = states[i];
        const auto& right = states[i + 1];
        if (station >= left.station && station <= right.station) {
            const double span = right.station - left.station;
            const double alpha = span <= 1.0e-12 ? 0.0 : (station - left.station) / span;
            return lerp(getter(left), getter(right), alpha);
        }
    }

    return getter(states.back());
}

double trapezoidalAverage(const std::vector<BranchSectionState>& states, const auto& getter) {
    if (states.empty()) {
        throw std::runtime_error("Branch requires at least one measured section station");
    }

    if (states.size() == 1U) {
        return getter(states.front());
    }

    double weighted_sum = 0.0;
    double span = states.back().station - states.front().station;
    if (span <= 0.0) {
        span = 1.0;
    }

    for (std::size_t i = 0; i + 1 < states.size(); ++i) {
        const auto& left = states[i];
        const auto& right = states[i + 1];
        const double ds = right.station - left.station;
        weighted_sum += 0.5 * (getter(left) + getter(right)) * ds;
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

BranchAverageProperties BranchComponent::reportAverageProperties(const MaterialLibrary& materials) const {
    const auto states = evaluateProfileStates(section_series_, materials);

    BranchAverageProperties result {};
    result.length = path_.length();
    result.average_area = trapezoidalAverage(states, [](const auto& state) { return state.area; });
    result.average_ix = trapezoidalAverage(states, [](const auto& state) { return state.ix; });
    result.average_iy = trapezoidalAverage(states, [](const auto& state) { return state.iy; });
    result.average_polar_moment = trapezoidalAverage(states, [](const auto& state) { return state.polar_moment; });
    result.average_mass_per_length = trapezoidalAverage(states, [](const auto& state) { return state.mass_per_length; });
    result.average_youngs_modulus = trapezoidalAverage(states, [](const auto& state) { return state.effective_youngs_modulus; });
    result.average_shear_modulus = trapezoidalAverage(states, [](const auto& state) { return state.effective_shear_modulus; });
    result.average_damping_ratio = trapezoidalAverage(states, [](const auto& state) { return state.damping_ratio; });

    return result;
}

BranchSectionState BranchComponent::evaluateSectionState(const MaterialLibrary& materials, const double station) const {
    const auto states = evaluateProfileStates(section_series_, materials);

    BranchSectionState result {};
    result.station = std::clamp(station, 0.0, 1.0);
    result.area = interpolateStateValue(states, result.station, [](const auto& state) { return state.area; });
    result.ix = interpolateStateValue(states, result.station, [](const auto& state) { return state.ix; });
    result.iy = interpolateStateValue(states, result.station, [](const auto& state) { return state.iy; });
    result.polar_moment = interpolateStateValue(states, result.station, [](const auto& state) { return state.polar_moment; });
    result.mass_per_length = interpolateStateValue(states, result.station, [](const auto& state) { return state.mass_per_length; });
    result.effective_youngs_modulus = interpolateStateValue(states, result.station, [](const auto& state) { return state.effective_youngs_modulus; });
    result.effective_shear_modulus = interpolateStateValue(states, result.station, [](const auto& state) { return state.effective_shear_modulus; });
    result.effective_poisson_ratio = interpolateStateValue(states, result.station, [](const auto& state) { return state.effective_poisson_ratio; });
    result.damping_ratio = interpolateStateValue(states, result.station, [](const auto& state) { return state.damping_ratio; });

    return result;
}

} // namespace orchard
