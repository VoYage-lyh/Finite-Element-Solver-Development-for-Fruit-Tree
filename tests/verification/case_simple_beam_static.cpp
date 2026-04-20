#include <iostream>

#include "common.h"

int main() {
    using namespace verification;

    constexpr double length = 1.0;
    constexpr double youngs_modulus = 1.0e10;
    constexpr double density = 800.0;
    constexpr double radius = 0.02;
    constexpr double area = 3.14159265358979323846 * radius * radius;
    constexpr double inertia = 3.14159265358979323846 * radius * radius * radius * radius / 4.0;
    constexpr double force = 100.0;

    const int num_elements = 20;
    const auto system = buildUniformPlanarBeam(num_elements, length, youngs_modulus, density, area, inertia);
    std::vector<double> load(system.stiffness.size(), 0.0);
    const int mid_node = num_elements / 2;
    load[static_cast<std::size_t>(2 * mid_node)] = force;

    const auto displacement = solveStaticSystem(system.stiffness, load, {0, 2 * num_elements});
    const double midspan_deflection = displacement[static_cast<std::size_t>(2 * mid_node)];
    const double expected = force * std::pow(length, 3) / (48.0 * youngs_modulus * inertia);

    checkClose(midspan_deflection, expected, expected * 1.0e-3, "simply supported midspan deflection should match analytic solution");
    std::cout << "midspan deflection=" << midspan_deflection << " expected=" << expected << '\n';
    return 0;
}
