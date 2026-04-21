#pragma once

#include <string>
#include <vector>

#include "orchard_solver/excitation_and_bc/Excitation.h"
#include "orchard_solver/solver_core/LinearAlgebra.h"

namespace orchard {

enum class NonlinearLinkKind {
    CubicSpring,
    GapSpring
};

struct NonlinearLink {
    std::string label;
    int first_dof {-1};
    int second_dof {-1};
    NonlinearLinkKind kind {NonlinearLinkKind::CubicSpring};
    double linear_stiffness {0.0};
    double cubic_stiffness {0.0};
    double open_stiffness {0.0};
    double gap_threshold {0.0};

    [[nodiscard]] double nonlinearForce(double relative_displacement) const;
    [[nodiscard]] double nonlinearTangent(double relative_displacement) const;
};

struct FrequencyResponsePoint {
    double frequency_hz {0.0};
    double excitation_response_magnitude {0.0};
    std::vector<double> observation_magnitudes;
};

struct FrequencyResponseResult {
    std::vector<std::string> observation_names;
    std::vector<FrequencyResponsePoint> points;

    void writeCsv(const std::string& file_path) const;
};

struct TimeHistoryPoint {
    double time_seconds {0.0};
    double excitation_signal_value {0.0};
    double excitation_load_value {0.0};
    double excitation_response_value {0.0};
    std::vector<double> observation_values;
};

struct TimeHistoryResult {
    std::vector<std::string> observation_names;
    std::vector<TimeHistoryPoint> points;

    void writeCsv(const std::string& file_path) const;
};

class DynamicSystem {
public:
    DenseMatrix mass;
    DenseMatrix damping;
    DenseMatrix stiffness;
    std::vector<std::string> dof_labels;
    std::vector<double> gravity_load;
    std::vector<NonlinearLink> nonlinear_links;
};

class FrequencyResponseAnalyzer {
public:
    [[nodiscard]] FrequencyResponseResult analyze(
        const DynamicSystem& system,
        int excitation_dof,
        const HarmonicExcitation& excitation,
        const AnalysisSettings& analysis,
        const std::vector<std::string>& observation_names,
        const std::vector<int>& observation_dofs
    ) const;
};

class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    [[nodiscard]] virtual TimeHistoryResult analyze(
        const DynamicSystem& system,
        int excitation_dof,
        const HarmonicExcitation& excitation,
        const AnalysisSettings& analysis,
        const std::vector<std::string>& observation_names,
        const std::vector<int>& observation_dofs
    ) const = 0;
};

class NewmarkIntegrator final : public TimeIntegrator {
public:
    [[nodiscard]] TimeHistoryResult analyze(
        const DynamicSystem& system,
        int excitation_dof,
        const HarmonicExcitation& excitation,
        const AnalysisSettings& analysis,
        const std::vector<std::string>& observation_names,
        const std::vector<int>& observation_dofs
    ) const override;
};

} // namespace orchard
