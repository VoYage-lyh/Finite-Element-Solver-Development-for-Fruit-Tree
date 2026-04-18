#pragma once

#include <string>

namespace orchard {

enum class AnalysisMode {
    FrequencyResponse,
    TimeHistory
};

enum class ExcitationKind {
    HarmonicForce,
    HarmonicDisplacement,
    HarmonicAcceleration
};

struct ClampBoundaryCondition {
    std::string branch_id;
    double support_stiffness {0.0};
    double support_damping {0.0};
    double cubic_stiffness {0.0};
};

struct HarmonicExcitation {
    ExcitationKind kind {ExcitationKind::HarmonicForce};
    std::string target_branch_id;
    double amplitude {0.0};
    double phase_degrees {0.0};
    double driving_frequency_hz {0.0};
};

struct AnalysisSettings {
    AnalysisMode mode {AnalysisMode::FrequencyResponse};
    double frequency_start_hz {1.0};
    double frequency_end_hz {25.0};
    int frequency_steps {50};
    double time_step_seconds {0.002};
    double total_time_seconds {1.0};
    int output_stride {1};
    int max_nonlinear_iterations {12};
    double nonlinear_tolerance {1.0e-8};
    std::string output_csv {"frequency_response.csv"};
};

} // namespace orchard
