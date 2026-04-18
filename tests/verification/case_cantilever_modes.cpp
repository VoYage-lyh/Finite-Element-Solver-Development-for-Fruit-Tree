#include <iostream>
#include <vector>

#include "common.h"

int main() {
    using namespace verification;

    constexpr double length = 1.0;
    constexpr double radius = 0.02;
    constexpr double youngs_modulus = 1.0e10;
    constexpr double density = 800.0;
    constexpr double area = 3.14159265358979323846 * radius * radius;
    constexpr double inertia = 3.14159265358979323846 * std::pow(radius, 4) / 4.0;
    const std::vector<double> betas = {
        1.875104068711961,
        4.694091132974175,
        7.854757438237613
    };

    std::vector<double> previous_errors;
    for (const int num_elements : {2, 4, 8, 16, 32}) {
        const auto system = buildUniformPlanarBeam(num_elements, length, youngs_modulus, density, area, inertia);
        const auto frequencies = generalizedFrequencies(system.stiffness, system.mass, {0, 1}, 3);

        std::vector<double> relative_errors;
        for (int mode = 0; mode < 3; ++mode) {
            const double expected_omega = betas[static_cast<std::size_t>(mode)] * betas[static_cast<std::size_t>(mode)] * std::sqrt((youngs_modulus * inertia) / (density * area * std::pow(length, 4)));
            const double expected_frequency = expected_omega / (2.0 * 3.14159265358979323846);
            relative_errors.push_back(std::abs(frequencies[static_cast<std::size_t>(mode)] - expected_frequency) / expected_frequency);
        }

        const double error_norm = l2Norm(relative_errors);
        std::cout << "num_elements=" << num_elements << " errors=("
                  << relative_errors[0] << ", "
                  << relative_errors[1] << ", "
                  << relative_errors[2] << ")\n";

        if (!previous_errors.empty()) {
            const double previous_norm = l2Norm(previous_errors);
            check(previous_norm / error_norm >= 3.0, "cantilever modal convergence rate should improve by at least 3x on each refinement");
        }
        previous_errors = relative_errors;

        if (num_elements == 32) {
            for (const auto relative_error : relative_errors) {
                check(relative_error < 0.005, "cantilever mode relative error at 32 elements must be below 0.5%");
            }
        }
    }

    return 0;
}
