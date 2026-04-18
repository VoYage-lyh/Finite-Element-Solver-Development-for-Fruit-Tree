#include <iostream>

#include "common.h"

namespace {

verification::PlanarBeamSystem buildHingedSystem(
    const double first_length,
    const double second_length,
    const double youngs_modulus,
    const double density,
    const double area,
    const double inertia,
    const double rotational_stiffness,
    const double tip_mass
) {
    verification::PlanarBeamSystem system {verification::zeros(7), verification::zeros(7)};

    const auto first = verification::buildUniformPlanarBeam(1, first_length, youngs_modulus, density, area, inertia);
    const auto second = verification::buildUniformPlanarBeam(1, second_length, youngs_modulus, density, area, inertia);

    const std::array<int, 4> first_map = {0, 1, 2, 3};
    const std::array<int, 4> second_map = {2, 4, 5, 6};

    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            system.stiffness[static_cast<std::size_t>(first_map[row])][static_cast<std::size_t>(first_map[col])] += first.stiffness[row][col];
            system.mass[static_cast<std::size_t>(first_map[row])][static_cast<std::size_t>(first_map[col])] += first.mass[row][col];
            system.stiffness[static_cast<std::size_t>(second_map[row])][static_cast<std::size_t>(second_map[col])] += second.stiffness[row][col];
            system.mass[static_cast<std::size_t>(second_map[row])][static_cast<std::size_t>(second_map[col])] += second.mass[row][col];
        }
    }

    system.stiffness[3][3] += rotational_stiffness;
    system.stiffness[3][4] -= rotational_stiffness;
    system.stiffness[4][3] -= rotational_stiffness;
    system.stiffness[4][4] += rotational_stiffness;
    system.mass[5][5] += tip_mass;

    return system;
}

} // namespace

int main() {
    using namespace verification;

    constexpr double first_length = 1.0;
    constexpr double second_length = 1.0;
    constexpr double youngs_modulus = 1.0e15;
    constexpr double density = 1.0e-9;
    constexpr double area = 1.0e-2;
    constexpr double inertia = 1.0e-4;
    constexpr double rotational_stiffness = 500.0;
    constexpr double tip_mass = 1.0;

    const auto system = buildHingedSystem(first_length, second_length, youngs_modulus, density, area, inertia, rotational_stiffness, tip_mass);
    const auto frequencies = generalizedFrequencies(system.stiffness, system.mass, {0, 1}, 1);
    const double expected = std::sqrt(rotational_stiffness / (tip_mass * second_length * second_length)) / (2.0 * 3.14159265358979323846);

    checkClose(frequencies.front(), expected, expected * 0.03, "hinged two-bar first frequency should match rigid-link spring-mass estimate");
    std::cout << "hinged two-bar first frequency=" << frequencies.front() << " expected=" << expected << '\n';
    return 0;
}
