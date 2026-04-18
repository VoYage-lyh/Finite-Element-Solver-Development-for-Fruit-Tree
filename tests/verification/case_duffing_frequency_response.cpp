#include <iostream>

#include "common.h"

int main() {
    using namespace verification;

    orchard::DynamicSystem system;
    system.mass = orchard::DenseMatrix(1, 1);
    system.damping = orchard::DenseMatrix(1, 1);
    system.stiffness = orchard::DenseMatrix(1, 1);
    system.mass(0, 0) = 1.0;
    system.damping(0, 0) = 0.5;
    system.stiffness(0, 0) = 100.0;
    system.dof_labels = {"duffing"};
    system.nonlinear_links.push_back(orchard::NonlinearLink {"duffing_ground", 0, -1, orchard::NonlinearLinkKind::CubicSpring, 100.0, 2.0e4, 0.0, 0.0});

    orchard::HarmonicExcitation excitation;
    excitation.kind = orchard::ExcitationKind::HarmonicForce;
    excitation.target_branch_id = "duffing";
    excitation.target_component = "ux";
    excitation.amplitude = 0.2;

    orchard::AnalysisSettings analysis;
    analysis.mode = orchard::AnalysisMode::TimeHistory;
    analysis.time_step_seconds = 0.001;
    analysis.total_time_seconds = 25.0;
    analysis.output_stride = 10;
    analysis.max_nonlinear_iterations = 50;
    analysis.nonlinear_tolerance = 1.0e-8;

    orchard::NewmarkIntegrator integrator;

    const double linear_frequency = std::sqrt(100.0) / (2.0 * 3.14159265358979323846);
    double peak_frequency = 0.0;
    double peak_amplitude = 0.0;
    for (int step = 0; step <= 20; ++step) {
        const double frequency = 1.35 + (0.04 * static_cast<double>(step));
        excitation.driving_frequency_hz = frequency;
        orchard::TimeHistoryResult response;
        try {
            response = integrator.analyze(system, 0, excitation, analysis, {"x"}, {0});
        } catch (const std::exception& exception) {
            throw std::runtime_error(
                "Duffing verification failed at frequency " + std::to_string(frequency) + " Hz: " + exception.what()
            );
        }

        const double amplitude = estimateSteadyAmplitude(response);
        if (amplitude > peak_amplitude) {
            peak_amplitude = amplitude;
            peak_frequency = frequency;
        }
    }

    check(peak_amplitude > 0.0, "Duffing sweep must produce a non-zero steady amplitude");
    const double predicted_peak =
        std::sqrt((100.0 + (0.75 * 2.0e4 * peak_amplitude * peak_amplitude)) / 1.0) / (2.0 * 3.14159265358979323846);
    const double relative_error = std::abs(peak_frequency - predicted_peak) / predicted_peak;

    check(relative_error < 0.05, "Duffing peak frequency should agree with backbone estimate within 5%");
    check(peak_frequency > linear_frequency * 1.03, "Duffing hardening should shift the peak frequency above the linear resonance");
    std::cout << "duffing peak=" << peak_frequency
              << " Hz predicted=" << predicted_peak
              << " Hz linear=" << linear_frequency
              << " Hz amplitude=" << peak_amplitude
              << '\n';
    return 0;
}
