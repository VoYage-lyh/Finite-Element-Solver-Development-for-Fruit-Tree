#include <iostream>
#include <memory>

#include "common.h"
#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/branches/BranchModel.h"
#include "orchard_solver/cross_section/CrossSection.h"
#include "orchard_solver/discretization/Assembler.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

namespace {

orchard::OrchardModel buildCantileverModel(const int num_elements) {
    using namespace orchard;

    OrchardModel model;
    MaterialProperties material;
    material.id = "xylem";
    material.tissue = TissueType::Xylem;
    material.density = 750.0;
    material.youngs_modulus = 1.0e10;
    material.poisson_ratio = 0.30;
    material.damping_ratio = 0.005;
    model.materials.addLinearElastic(material);

    BranchPath path(Vec3 {0.0, 0.0, 0.0}, Vec3 {1.0, 0.0, 0.0});
    model.topology.addBranch("cantilever", std::nullopt, 0, path);
    model.topology.rebuildChildLinks();

    orchard::RegionGeometry geometry;
    geometry.kind = orchard::SectionShapeKind::SolidEllipse;
    geometry.center = orchard::Vec2 {0.0, 0.0};
    geometry.radii = orchard::Vec2 {0.02, 0.02};
    geometry.samples = 96;

    std::vector<orchard::TissueRegionDefinition> regions {
        orchard::TissueRegionDefinition {orchard::TissueType::Xylem, "xylem", geometry}
    };

    orchard::MeasuredSectionSeries section_series;
    section_series.addProfile(std::make_shared<orchard::ParameterizedSectionProfile>(0.0, regions));
    section_series.addProfile(std::make_shared<orchard::ParameterizedSectionProfile>(1.0, regions));

    model.branches.emplace_back("cantilever", std::nullopt, 0, path, std::move(section_series), orchard::BranchDiscretizationHint {num_elements, false});
    model.clamps.push_back(orchard::ClampBoundaryCondition {"cantilever", 0.0, 0.0, 0.0});
    model.excitation.kind = orchard::ExcitationKind::HarmonicForce;
    model.excitation.target_branch_id = "cantilever";
    model.excitation.target_node = "tip";
    model.excitation.target_component = "uy";
    model.excitation.amplitude = 1.0;
    model.analysis.frequency_start_hz = 1.0;
    model.analysis.frequency_end_hz = 50.0;
    model.analysis.frequency_steps = 240;
    model.analysis.rayleigh_alpha = 0.0;
    model.analysis.rayleigh_beta = 0.0;
    model.observations.push_back(orchard::ObservationPoint {"tip_uy", "branch", "cantilever", "tip", "uy"});
    return model;
}

} // namespace

int main() {
    using namespace verification;

    const auto model = buildCantileverModel(10);
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

    const double radius = 0.02;
    const double area = 3.14159265358979323846 * radius * radius;
    const double inertia = 3.14159265358979323846 * std::pow(radius, 4) / 4.0;
    constexpr double beta_1 = 1.875104068711961;
    const double omega = std::pow(beta_1, 2) * std::sqrt((1.0e10 * inertia) / (750.0 * area));
    const double expected_frequency = omega / (2.0 * 3.14159265358979323846);

    checkClose(peak_frequency, expected_frequency, expected_frequency * 0.05, "cantilever first frequency should match analytic reference");
    std::cout << "cantilever first mode peak=" << peak_frequency << " Hz expected=" << expected_frequency << " Hz\n";
    return 0;
}
