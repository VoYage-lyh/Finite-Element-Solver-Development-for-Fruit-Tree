#include "orchard_solver/solver_core/DynamicSystem.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace orchard {

namespace {

constexpr double kPi = 3.14159265358979323846;

double defaultDrivingFrequencyHz(const HarmonicExcitation& excitation, const AnalysisSettings& analysis) {
    if (excitation.driving_frequency_hz > 0.0) {
        return excitation.driving_frequency_hz;
    }

    return std::max(analysis.frequency_start_hz, 0.1);
}

std::complex<double> buildFrequencyExcitationLoad(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const double omega
) {
    const double phase_radians = excitation.phase_degrees * (kPi / 180.0);
    const std::complex<double> base = std::polar(excitation.amplitude, phase_radians);

    switch (excitation.kind) {
    case ExcitationKind::HarmonicForce:
        return base;
    case ExcitationKind::HarmonicDisplacement:
        return base * std::complex<double>(system.stiffness(excitation_dof, excitation_dof), omega * system.damping(excitation_dof, excitation_dof));
    case ExcitationKind::HarmonicAcceleration:
        return base * system.mass(excitation_dof, excitation_dof);
    }

    throw std::runtime_error("Unsupported excitation kind");
}

double buildTimeExcitationLoad(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const AnalysisSettings& analysis,
    const double time_seconds
) {
    const double phase_radians = excitation.phase_degrees * (kPi / 180.0);
    const double omega = 2.0 * kPi * defaultDrivingFrequencyHz(excitation, analysis);
    const double angle = (omega * time_seconds) + phase_radians;
    const double displacement = excitation.amplitude * std::sin(angle);
    const double velocity = excitation.amplitude * omega * std::cos(angle);
    const double acceleration = -excitation.amplitude * omega * omega * std::sin(angle);

    switch (excitation.kind) {
    case ExcitationKind::HarmonicForce:
        return displacement;
    case ExcitationKind::HarmonicDisplacement:
        return (system.stiffness(excitation_dof, excitation_dof) * displacement)
            + (system.damping(excitation_dof, excitation_dof) * velocity)
            + (system.mass(excitation_dof, excitation_dof) * acceleration);
    case ExcitationKind::HarmonicAcceleration:
        return system.mass(excitation_dof, excitation_dof) * acceleration;
    }

    throw std::runtime_error("Unsupported excitation kind");
}

std::vector<double> buildLoadVector(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const AnalysisSettings& analysis,
    const double time_seconds
) {
    std::vector<double> load(system.mass.rows(), 0.0);
    load.at(static_cast<std::size_t>(excitation_dof)) = buildTimeExcitationLoad(system, excitation_dof, excitation, analysis, time_seconds);
    return load;
}

std::vector<double> multiply(const DenseMatrix& matrix, const std::vector<double>& vector) {
    std::vector<double> result(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result[row] += matrix(row, col) * vector[col];
        }
    }
    return result;
}

double infinityNorm(const std::vector<double>& values) {
    double result = 0.0;
    for (const auto value : values) {
        result = std::max(result, std::abs(value));
    }
    return result;
}

DenseMatrix evaluateNonlinearTangentAndForce(
    const DynamicSystem& system,
    const std::vector<double>& displacement,
    std::vector<double>& nonlinear_force
) {
    const std::size_t dof_count = system.mass.rows();
    DenseMatrix tangent(dof_count, dof_count);
    nonlinear_force.assign(dof_count, 0.0);

    for (const auto& link : system.nonlinear_links) {
        const int first = link.first_dof;
        const int second = link.second_dof;
        const double second_value = second >= 0 ? displacement.at(static_cast<std::size_t>(second)) : 0.0;
        const double relative_displacement = displacement.at(static_cast<std::size_t>(first)) - second_value;
        const double scalar_force = link.nonlinearForce(relative_displacement);
        const double scalar_tangent = link.nonlinearTangent(relative_displacement);

        nonlinear_force.at(static_cast<std::size_t>(first)) += scalar_force;
        tangent.add(static_cast<std::size_t>(first), static_cast<std::size_t>(first), scalar_tangent);

        if (second >= 0) {
            nonlinear_force.at(static_cast<std::size_t>(second)) -= scalar_force;
            tangent.add(static_cast<std::size_t>(first), static_cast<std::size_t>(second), -scalar_tangent);
            tangent.add(static_cast<std::size_t>(second), static_cast<std::size_t>(first), -scalar_tangent);
            tangent.add(static_cast<std::size_t>(second), static_cast<std::size_t>(second), scalar_tangent);
        }
    }

    return tangent;
}

std::vector<std::vector<double>> buildEffectiveMatrix(
    const DynamicSystem& system,
    const DenseMatrix& nonlinear_tangent,
    const double mass_scale,
    const double damping_scale
) {
    std::vector<std::vector<double>> matrix(
        system.mass.rows(),
        std::vector<double>(system.mass.cols(), 0.0)
    );

    for (std::size_t row = 0; row < system.mass.rows(); ++row) {
        for (std::size_t col = 0; col < system.mass.cols(); ++col) {
            matrix[row][col] =
                (mass_scale * system.mass(row, col))
                + (damping_scale * system.damping(row, col))
                + system.stiffness(row, col)
                + nonlinear_tangent(row, col);
        }
    }

    return matrix;
}

std::vector<double> computeInitialAcceleration(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const AnalysisSettings& analysis
) {
    const auto external = buildLoadVector(system, excitation_dof, excitation, analysis, 0.0);
    std::vector<double> nonlinear_force;
    const auto nonlinear_tangent = evaluateNonlinearTangentAndForce(
        system,
        std::vector<double>(system.mass.rows(), 0.0),
        nonlinear_force
    );
    (void)nonlinear_tangent;

    std::vector<double> rhs = external;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] -= nonlinear_force[i];
    }

    std::vector<std::vector<double>> mass_matrix(system.mass.rows(), std::vector<double>(system.mass.cols(), 0.0));
    for (std::size_t row = 0; row < system.mass.rows(); ++row) {
        for (std::size_t col = 0; col < system.mass.cols(); ++col) {
            mass_matrix[row][col] = system.mass(row, col);
        }
    }

    return solveLinearSystem(std::move(mass_matrix), std::move(rhs));
}

} // namespace

double NonlinearLink::nonlinearForce(const double relative_displacement) const {
    switch (kind) {
    case NonlinearLinkKind::CubicSpring:
        return cubic_stiffness * relative_displacement * relative_displacement * relative_displacement;
    case NonlinearLinkKind::GapSpring: {
        const double magnitude = std::abs(relative_displacement);
        if (magnitude <= gap_threshold) {
            return 0.0;
        }

        return std::copysign((open_stiffness - linear_stiffness) * (magnitude - gap_threshold), relative_displacement);
    }
    }

    throw std::runtime_error("Unsupported nonlinear link kind");
}

double NonlinearLink::nonlinearTangent(const double relative_displacement) const {
    switch (kind) {
    case NonlinearLinkKind::CubicSpring:
        return 3.0 * cubic_stiffness * relative_displacement * relative_displacement;
    case NonlinearLinkKind::GapSpring:
        return std::abs(relative_displacement) <= gap_threshold ? 0.0 : (open_stiffness - linear_stiffness);
    }

    throw std::runtime_error("Unsupported nonlinear link kind");
}

void FrequencyResponseResult::writeCsv(const std::string& file_path) const {
    std::ofstream stream(file_path);
    if (!stream) {
        throw std::runtime_error("Unable to open output CSV: " + file_path);
    }

    stream << "frequency_hz";
    for (const auto& name : observation_names) {
        stream << ',' << name;
    }
    stream << '\n';

    stream << std::setprecision(10);
    for (const auto& point : points) {
        stream << point.frequency_hz;
        for (const auto value : point.observation_magnitudes) {
            stream << ',' << value;
        }
        stream << '\n';
    }
}

void TimeHistoryResult::writeCsv(const std::string& file_path) const {
    std::ofstream stream(file_path);
    if (!stream) {
        throw std::runtime_error("Unable to open output CSV: " + file_path);
    }

    stream << "time_s";
    for (const auto& name : observation_names) {
        stream << ',' << name;
    }
    stream << '\n';

    stream << std::setprecision(10);
    for (const auto& point : points) {
        stream << point.time_seconds;
        for (const auto value : point.observation_values) {
            stream << ',' << value;
        }
        stream << '\n';
    }
}

FrequencyResponseResult FrequencyResponseAnalyzer::analyze(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const AnalysisSettings& analysis,
    const std::vector<std::string>& observation_names,
    const std::vector<int>& observation_dofs
) const {
    if (system.mass.rows() == 0U) {
        throw std::runtime_error("Dynamic system is empty");
    }

    const std::size_t dof_count = system.mass.rows();
    const int steps = std::max(analysis.frequency_steps, 1);

    FrequencyResponseResult result;
    result.observation_names = observation_names;
    result.points.reserve(static_cast<std::size_t>(steps));

    for (int step = 0; step < steps; ++step) {
        const double alpha = steps == 1 ? 0.0 : static_cast<double>(step) / static_cast<double>(steps - 1);
        const double frequency_hz = analysis.frequency_start_hz + alpha * (analysis.frequency_end_hz - analysis.frequency_start_hz);
        const double omega = 2.0 * kPi * frequency_hz;

        std::vector<std::vector<std::complex<double>>> dynamic_stiffness(
            dof_count,
            std::vector<std::complex<double>>(dof_count, std::complex<double> {0.0, 0.0})
        );
        std::vector<std::complex<double>> load(dof_count, std::complex<double> {0.0, 0.0});

        for (std::size_t row = 0; row < dof_count; ++row) {
            for (std::size_t col = 0; col < dof_count; ++col) {
                dynamic_stiffness[row][col] = std::complex<double>(
                    system.stiffness(row, col) - ((omega * omega) * system.mass(row, col)),
                    omega * system.damping(row, col)
                );
            }
        }

        load[static_cast<std::size_t>(excitation_dof)] = buildFrequencyExcitationLoad(system, excitation_dof, excitation, omega);
        const auto response = solveComplexLinearSystem(std::move(dynamic_stiffness), std::move(load));

        FrequencyResponsePoint point;
        point.frequency_hz = frequency_hz;
        point.observation_magnitudes.reserve(observation_dofs.size());
        for (const auto observation_dof : observation_dofs) {
            point.observation_magnitudes.push_back(std::abs(response.at(static_cast<std::size_t>(observation_dof))));
        }

        result.points.push_back(std::move(point));
    }

    return result;
}

TimeHistoryResult NewmarkIntegrator::analyze(
    const DynamicSystem& system,
    const int excitation_dof,
    const HarmonicExcitation& excitation,
    const AnalysisSettings& analysis,
    const std::vector<std::string>& observation_names,
    const std::vector<int>& observation_dofs
) const {
    if (system.mass.rows() == 0U) {
        throw std::runtime_error("Dynamic system is empty");
    }
    if (analysis.time_step_seconds <= 0.0 || analysis.total_time_seconds <= 0.0) {
        throw std::runtime_error("Time-history analysis requires positive time step and total time");
    }

    const std::size_t dof_count = system.mass.rows();
    const double dt = analysis.time_step_seconds;
    const int total_steps = std::max(1, static_cast<int>(std::round(analysis.total_time_seconds / dt)));
    const int output_stride = std::max(analysis.output_stride, 1);
    const double beta = 0.25;
    const double gamma = 0.5;
    const double mass_scale = 1.0 / (beta * dt * dt);
    const double damping_scale = gamma / (beta * dt);

    std::vector<double> displacement(dof_count, 0.0);
    std::vector<double> velocity(dof_count, 0.0);
    std::vector<double> acceleration = computeInitialAcceleration(system, excitation_dof, excitation, analysis);

    TimeHistoryResult result;
    result.observation_names = observation_names;
    result.points.reserve(static_cast<std::size_t>((total_steps / output_stride) + 1));
    result.points.push_back(TimeHistoryPoint {0.0, std::vector<double>(observation_dofs.size(), 0.0)});

    for (int step = 1; step <= total_steps; ++step) {
        const double time_seconds = static_cast<double>(step) * dt;

        std::vector<double> displacement_predictor(dof_count, 0.0);
        std::vector<double> velocity_predictor(dof_count, 0.0);
        for (std::size_t i = 0; i < dof_count; ++i) {
            displacement_predictor[i] = displacement[i] + (dt * velocity[i]) + (dt * dt * (0.5 - beta) * acceleration[i]);
            velocity_predictor[i] = velocity[i] + (dt * (1.0 - gamma) * acceleration[i]);
        }

        std::vector<double> displacement_guess = displacement_predictor;
        bool converged = false;

        for (int iteration = 0; iteration < std::max(analysis.max_nonlinear_iterations, 1); ++iteration) {
            std::vector<double> acceleration_guess(dof_count, 0.0);
            std::vector<double> velocity_guess(dof_count, 0.0);
            for (std::size_t i = 0; i < dof_count; ++i) {
                acceleration_guess[i] = mass_scale * (displacement_guess[i] - displacement_predictor[i]);
                velocity_guess[i] = velocity_predictor[i] + (gamma * dt * acceleration_guess[i]);
            }

            std::vector<double> nonlinear_force;
            const DenseMatrix nonlinear_tangent = evaluateNonlinearTangentAndForce(system, displacement_guess, nonlinear_force);
            const auto external_force = buildLoadVector(system, excitation_dof, excitation, analysis, time_seconds);
            const auto mass_force = multiply(system.mass, acceleration_guess);
            const auto damping_force = multiply(system.damping, velocity_guess);
            const auto stiffness_force = multiply(system.stiffness, displacement_guess);

            std::vector<double> residual(dof_count, 0.0);
            for (std::size_t i = 0; i < dof_count; ++i) {
                residual[i] = mass_force[i] + damping_force[i] + stiffness_force[i] + nonlinear_force[i] - external_force[i];
            }

            const double residual_norm = infinityNorm(residual);
            if (residual_norm < analysis.nonlinear_tolerance) {
                converged = true;
                break;
            }

            auto effective_matrix = buildEffectiveMatrix(system, nonlinear_tangent, mass_scale, damping_scale);
            std::vector<double> negative_residual(dof_count, 0.0);
            for (std::size_t i = 0; i < dof_count; ++i) {
                negative_residual[i] = -residual[i];
            }
            const auto displacement_increment = solveLinearSystem(std::move(effective_matrix), std::move(negative_residual));

            for (std::size_t i = 0; i < dof_count; ++i) {
                displacement_guess[i] += displacement_increment[i];
            }

            const double increment_norm = infinityNorm(displacement_increment);
            const double state_norm = std::max(infinityNorm(displacement_guess), 1.0);
            if (increment_norm < (analysis.nonlinear_tolerance * state_norm)) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            throw std::runtime_error("Newmark nonlinear iteration failed to converge at time " + std::to_string(time_seconds));
        }

        for (std::size_t i = 0; i < dof_count; ++i) {
            acceleration[i] = mass_scale * (displacement_guess[i] - displacement_predictor[i]);
            velocity[i] = velocity_predictor[i] + (gamma * dt * acceleration[i]);
            displacement[i] = displacement_guess[i];
        }

        if (step % output_stride == 0 || step == total_steps) {
            TimeHistoryPoint point;
            point.time_seconds = time_seconds;
            point.observation_values.reserve(observation_dofs.size());
            for (const auto observation_dof : observation_dofs) {
                point.observation_values.push_back(displacement.at(static_cast<std::size_t>(observation_dof)));
            }
            result.points.push_back(std::move(point));
        }
    }

    return result;
}

} // namespace orchard
