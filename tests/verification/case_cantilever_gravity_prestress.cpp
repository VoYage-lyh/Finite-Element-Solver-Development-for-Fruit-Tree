#include <iostream>
#include <memory>

#include "common.h"
#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/branches/BranchModel.h"
#include "orchard_solver/cross_section/CrossSection.h"
#include "orchard_solver/discretization/Assembler.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

namespace {

orchard::OrchardModel buildVerticalCantileverModel(const bool include_gravity_prestress) {
    using namespace orchard;

    constexpr double length = 1.5;
    constexpr double radius = 0.005;

    OrchardModel model;
    MaterialProperties material;
    material.id = "xylem";
    material.tissue = TissueType::Xylem;
    material.density = 750.0;
    material.youngs_modulus = 1.0e10;
    material.poisson_ratio = 0.30;
    material.damping_ratio = 0.002;
    model.materials.addLinearElastic(material);

    BranchPath path(Vec3 {0.0, 0.0, 0.0}, Vec3 {0.0, 0.0, length});
    model.topology.addBranch("cantilever", std::nullopt, 0, path);
    model.topology.rebuildChildLinks();

    RegionGeometry geometry;
    geometry.kind = SectionShapeKind::SolidEllipse;
    geometry.center = Vec2 {0.0, 0.0};
    geometry.radii = Vec2 {radius, radius};
    geometry.samples = 96;

    std::vector<TissueRegionDefinition> regions {
        TissueRegionDefinition {TissueType::Xylem, "xylem", geometry}
    };

    MeasuredSectionSeries section_series;
    section_series.addProfile(std::make_shared<ParameterizedSectionProfile>(0.0, regions));
    section_series.addProfile(std::make_shared<ParameterizedSectionProfile>(1.0, regions));

    model.branches.emplace_back(
        "cantilever",
        std::nullopt,
        0,
        path,
        std::move(section_series),
        BranchDiscretizationHint {16, false}
    );
    model.clamps.push_back(ClampBoundaryCondition {"cantilever", 0.0, 0.0, 0.0});
    model.excitation.kind = ExcitationKind::HarmonicForce;
    model.excitation.target_branch_id = "cantilever";
    model.excitation.target_node = "tip";
    model.excitation.target_component = "uy";
    model.excitation.amplitude = 1.0;
    model.analysis.frequency_start_hz = 0.5;
    model.analysis.frequency_end_hz = 4.5;
    model.analysis.frequency_steps = 500;
    model.analysis.rayleigh_alpha = 0.0;
    model.analysis.rayleigh_beta = 0.0;
    model.analysis.include_gravity_prestress = include_gravity_prestress;
    model.analysis.gravity_direction = Vec3 {0.0, 0.0, -1.0};
    model.observations.push_back(ObservationPoint {"tip_uy", "branch", "cantilever", "tip", {"uy"}});
    return model;
}

double findPeakFrequency(const orchard::OrchardModel& model) {
    orchard::StructuralAssembler assembler;
    const auto assembled = assembler.assemble(model);

    orchard::FrequencyResponseAnalyzer analyzer;
    const auto response = analyzer.analyze(
        assembled.system,
        assembled.excitation_dof,
        model.excitation,
        model.analysis,
        assembled.observation_names,
        assembled.observation_dofs
    );

    double peak_frequency = response.points.front().frequency_hz;
    double peak_magnitude = response.points.front().observation_magnitudes.front();
    for (const auto& point : response.points) {
        if (point.observation_magnitudes.front() > peak_magnitude) {
            peak_magnitude = point.observation_magnitudes.front();
            peak_frequency = point.frequency_hz;
        }
    }
    return peak_frequency;
}

} // namespace

int main() {
    using namespace verification;

    constexpr double pi = 3.14159265358979323846;
    constexpr double length = 1.5;
    constexpr double radius = 0.005;
    constexpr double density = 750.0;
    constexpr double youngs_modulus = 1.0e10;
    constexpr double gravity = 9.81;
    constexpr double beta_1 = 1.875104068711961;

    const auto baseline_model = buildVerticalCantileverModel(false);
    const auto prestressed_model = buildVerticalCantileverModel(true);

    const double baseline_frequency = findPeakFrequency(baseline_model);
    const double prestressed_frequency = findPeakFrequency(prestressed_model);

    const double area = pi * radius * radius;
    const double inertia = pi * std::pow(radius, 4) / 4.0;
    const double omega_0 = std::pow(beta_1, 2) * std::sqrt((youngs_modulus * inertia) / (density * area * std::pow(length, 4)));
    const double expected_baseline = omega_0 / (2.0 * pi);

    const double average_axial_force = density * area * gravity * length * 0.5;
    const double critical_load = (pi * pi * youngs_modulus * inertia) / (4.0 * length * length);
    const double expected_prestressed = expected_baseline * std::sqrt(1.0 - (average_axial_force / critical_load));

    checkClose(
        baseline_frequency,
        expected_baseline,
        expected_baseline * 0.05,
        "baseline cantilever frequency should match the Euler-Bernoulli reference"
    );
    check(prestressed_frequency < baseline_frequency, "gravity prestress should reduce the first bending frequency");
    checkClose(
        prestressed_frequency,
        expected_prestressed,
        expected_prestressed * 0.05,
        "prestressed cantilever frequency should match the geometric-stiffness estimate"
    );

    std::cout
        << "baseline=" << baseline_frequency
        << " Hz prestressed=" << prestressed_frequency
        << " Hz expected_prestressed=" << expected_prestressed
        << " Hz\n";
    return 0;
}
