#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/branches/BranchModel.h"
#include "orchard_solver/cross_section/CrossSection.h"
#include "orchard_solver/discretization/Assembler.h"
#include "orchard_solver/io/ModelIO.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

#ifndef ORCHARD_SOURCE_DIR
#define ORCHARD_SOURCE_DIR "."
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string exampleModelPath() {
    return std::string(ORCHARD_SOURCE_DIR) + "/examples/demo_orchard.json";
}

void check(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkClose(const double actual, const double expected, const double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

orchard::OrchardModel buildCantileverModel(const int num_elements, const double damping_ratio = 0.01) {
    using namespace orchard;

    OrchardModel model;
    model.metadata.name = "cantilever_beam_reference";

    MaterialProperties material;
    material.id = "cantilever_xylem";
    material.tissue = TissueType::Xylem;
    material.density = 750.0;
    material.youngs_modulus = 1.0e10;
    material.poisson_ratio = 0.30;
    material.damping_ratio = damping_ratio;
    model.materials.addLinearElastic(material);

    TreeTopology topology;
    BranchPath path(Vec3 {0.0, 0.0, 0.0}, Vec3 {0.0, 0.0, 1.0});
    topology.addBranch("cantilever", std::nullopt, 0, path);
    topology.rebuildChildLinks();
    model.topology = topology;

    RegionGeometry geometry;
    geometry.kind = SectionShapeKind::SolidEllipse;
    geometry.center = Vec2 {0.0, 0.0};
    geometry.radii = Vec2 {0.02, 0.02};
    geometry.samples = 96;

    std::vector<TissueRegionDefinition> regions {
        TissueRegionDefinition {TissueType::Xylem, "cantilever_xylem", geometry}
    };

    MeasuredSectionSeries series;
    series.addProfile(std::make_shared<ParameterizedSectionProfile>(0.0, regions));
    series.addProfile(std::make_shared<ParameterizedSectionProfile>(1.0, regions));

    model.branches.emplace_back(
        "cantilever",
        std::nullopt,
        0,
        path,
        std::move(series),
        BranchDiscretizationHint {num_elements, false}
    );

    model.clamps.push_back(ClampBoundaryCondition {"cantilever", 1.0, 0.0, 0.0});

    model.excitation.kind = ExcitationKind::HarmonicForce;
    model.excitation.target_branch_id = "cantilever";
    model.excitation.target_node = "tip";
    model.excitation.target_component = "ux";
    model.excitation.amplitude = 1.0;
    model.excitation.phase_degrees = 0.0;

    model.analysis.mode = AnalysisMode::FrequencyResponse;
    model.analysis.frequency_start_hz = 1.0;
    model.analysis.frequency_end_hz = 50.0;
    model.analysis.frequency_steps = 240;
    model.analysis.rayleigh_alpha = 0.0;
    model.analysis.rayleigh_beta = 0.0;

    model.observations.push_back(ObservationPoint {"tip_ux", "branch", "cantilever", "tip", {"ux"}});

    return model;
}

void testSectionPartitionGeometry() {
    using namespace orchard;

    std::vector<TissueRegionDefinition> regions;
    regions.push_back(TissueRegionDefinition {
        TissueType::Pith,
        "pith",
        RegionGeometry {SectionShapeKind::SolidEllipse, Vec2 {0.004, -0.002}, Vec2 {0.012, 0.009}, {}, {}, {}, {}, {}, {}, 96}
    });

    RegionGeometry xylem_geometry;
    xylem_geometry.kind = SectionShapeKind::EllipticRing;
    xylem_geometry.outer_center = Vec2 {0.0, 0.0};
    xylem_geometry.outer_radii = Vec2 {0.032, 0.024};
    xylem_geometry.inner_center = Vec2 {0.004, -0.002};
    xylem_geometry.inner_radii = Vec2 {0.012, 0.009};
    xylem_geometry.samples = 96;
    regions.push_back(TissueRegionDefinition {TissueType::Xylem, "xylem", xylem_geometry});

    RegionGeometry phloem_geometry;
    phloem_geometry.kind = SectionShapeKind::EllipticRing;
    phloem_geometry.outer_center = Vec2 {0.0, 0.0};
    phloem_geometry.outer_radii = Vec2 {0.037, 0.028};
    phloem_geometry.inner_center = Vec2 {0.0, 0.0};
    phloem_geometry.inner_radii = Vec2 {0.032, 0.024};
    phloem_geometry.samples = 96;
    regions.push_back(TissueRegionDefinition {TissueType::Phloem, "phloem", phloem_geometry});

    orchard::ParameterizedSectionProfile profile(0.0, regions);
    const auto properties = profile.evaluate();

    const double expected_pith = kPi * 0.012 * 0.009;
    const double expected_xylem = (kPi * 0.032 * 0.024) - expected_pith;
    const double expected_phloem = (kPi * 0.037 * 0.028) - (kPi * 0.032 * 0.024);
    const double expected_total = expected_pith + expected_xylem + expected_phloem;

    check(properties.regions.size() == 3U, "section should expose three tissue regions");
    checkClose(properties.total_area, expected_total, expected_total * 0.03, "section total area should match nested ellipse estimate");
    check(properties.ix_centroid > 0.0, "section Ix must be positive");
    check(properties.iy_centroid > 0.0, "section Iy must be positive");
}

void testMaterialLoading() {
    const auto model = orchard::loadModelFromFile(exampleModelPath());
    const auto ids = model.materials.ids();
    check(ids.size() >= 3U, "example model should load at least three materials");
    check(model.materials.contains("xylem_default"), "example model should load xylem_default");
    check(model.materials.contains("pith_default"), "example model should load pith_default");
    check(model.materials.contains("phloem_default"), "example model should load phloem_default");
}

void testTopologyAssembly() {
    const auto model = orchard::loadModelFromFile(exampleModelPath());
    const auto& trunk = model.topology.requireNode("trunk");
    const auto& secondary = model.topology.requireNode("secondary_left");

    check(!trunk.parent_branch_id.has_value(), "trunk should be root");
    check(trunk.child_branch_ids.size() == 2U, "trunk should have two child branches");
    check(secondary.parent_branch_id.has_value(), "secondary_left should have a parent");
    check(*secondary.parent_branch_id == "primary_left", "secondary_left should descend from primary_left");
}

void testMatrixAssembly() {
    const auto model = orchard::loadModelFromFile(exampleModelPath());
    orchard::StructuralAssembler assembler;
    const auto assembled = assembler.assemble(model);

    int expected_branch_dofs = 0;
    for (const auto& branch : model.branches) {
        expected_branch_dofs += 6 * (std::max(branch.discretizationHint().num_elements, 1) + 1);
    }
    const int expected_total_dofs = expected_branch_dofs + static_cast<int>(model.fruits.size());

    check(static_cast<int>(assembled.system.mass.rows()) == expected_total_dofs, "DOF count should equal 6*branch_nodes + fruit_count");
    check(assembled.system.mass.cols() == assembled.system.mass.rows(), "mass matrix should be square");
    check(!assembled.branch_nodes.at("trunk").empty(), "trunk should have discretized nodes");
    check(assembled.excitation_dof == assembled.requireBranchDof("trunk", static_cast<int>(assembled.branch_nodes.at("trunk").size()) - 1, "ux"), "excitation should target trunk tip ux DOF");
    check(assembled.system.mass(assembled.excitation_dof, assembled.excitation_dof) > 0.0, "tip translational mass must be positive");
}

void testDemoResponseAndCsvOutput() {
    const auto model = orchard::loadModelFromFile(exampleModelPath());
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

    check(static_cast<int>(response.points.size()) == model.analysis.frequency_steps, "response should contain configured number of frequency samples");
    check(!response.points.empty(), "response must not be empty");
    check(response.points.front().observation_magnitudes.size() == assembled.observation_names.size(), "response should contain one magnitude per observation");
    check(response.points.front().observation_magnitudes.front() >= 0.0, "response magnitude should be non-negative");

    const std::filesystem::path output_path = std::filesystem::current_path() / "orchard_test_frequency_response.csv";
    response.writeCsv(output_path.string());
    check(std::filesystem::exists(output_path), "frequency response CSV should be written");

    {
        std::ifstream stream(output_path);
        std::string header;
        std::getline(stream, header);
        check(header.find("frequency_hz") != std::string::npos, "CSV header should contain frequency_hz");
        check(header.find("excitation_response") != std::string::npos, "frequency response CSV should expose excitation_response");
    }
}

void testTimeHistoryCsvIncludesExcitationChannels() {
    auto model = buildCantileverModel(8, 0.01);
    model.analysis.mode = orchard::AnalysisMode::TimeHistory;
    model.analysis.time_step_seconds = 0.002;
    model.analysis.total_time_seconds = 0.12;
    model.analysis.output_stride = 2;
    model.analysis.max_nonlinear_iterations = 20;
    model.analysis.nonlinear_tolerance = 1.0e-8;
    model.excitation.driving_frequency_hz = 8.0;
    model.excitation.target_component = "ux";
    model.observations.clear();
    model.observations.push_back(orchard::ObservationPoint {"tip_ux", "branch", "cantilever", "tip", {"ux"}});

    orchard::StructuralAssembler assembler;
    const auto assembled = assembler.assemble(model);

    orchard::NewmarkIntegrator integrator;
    const auto response = integrator.analyze(
        assembled.system,
        assembled.excitation_dof,
        model.excitation,
        model.analysis,
        assembled.observation_names,
        assembled.observation_dofs
    );

    check(!response.points.empty(), "time-history response must not be empty");
    check(response.points.front().time_seconds == 0.0, "time-history response should include the initial sample");

    const std::filesystem::path output_path = std::filesystem::current_path() / "orchard_test_time_history.csv";
    response.writeCsv(output_path.string());
    check(std::filesystem::exists(output_path), "time-history CSV should be written");

    {
        std::ifstream stream(output_path);
        std::string header;
        std::getline(stream, header);
        check(header.find("time_s") != std::string::npos, "CSV header should contain time_s");
        check(header.find("excitation_signal") != std::string::npos, "time-history CSV should contain excitation_signal");
        check(header.find("excitation_load") != std::string::npos, "time-history CSV should contain excitation_load");
        check(header.find("excitation_response") != std::string::npos, "time-history CSV should contain excitation_response");
    }
}

void testCircularShorthandLoading() {
    const std::filesystem::path json_path = std::filesystem::current_path() / "orchard_test_circular_shorthand.json";
    {
        std::ofstream stream(json_path);
        stream << R"json(
{
  "metadata": {
    "name": "circular_shorthand_cpp"
  },
  "materials": [
    {
      "id": "xylem_default",
      "tissue": "xylem",
      "model": "linear",
      "density": 700.0,
      "youngs_modulus": 9000000000.0,
      "poisson_ratio": 0.3,
      "damping_ratio": 0.02
    },
    {
      "id": "pith_default",
      "tissue": "pith",
      "model": "linear",
      "density": 180.0,
      "youngs_modulus": 400000000.0,
      "poisson_ratio": 0.25,
      "damping_ratio": 0.04
    },
    {
      "id": "phloem_default",
      "tissue": "phloem",
      "model": "linear",
      "density": 950.0,
      "youngs_modulus": 150000000.0,
      "poisson_ratio": 0.35,
      "damping_ratio": 0.08
    }
  ],
  "branches": [
    {
      "id": "trunk",
      "parent_branch_id": null,
      "level": 0,
      "start": [0.0, 0.0, 0.0],
      "end": [0.0, 0.0, 1.0],
      "stations": [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.025},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.02}
      ],
      "discretization": {"num_elements": 2, "hotspot": false}
    }
  ],
  "fruits": [],
  "clamps": [
    {
      "branch_id": "trunk",
      "support_stiffness": 1.0,
      "support_damping": 0.0,
      "cubic_stiffness": 0.0
    }
  ],
  "excitation": {
    "kind": "harmonic_force",
    "target_branch_id": "trunk",
    "target_node": "tip",
    "target_component": "ux",
    "amplitude": 1.0,
    "phase_degrees": 0.0,
    "driving_frequency_hz": 5.0
  },
  "analysis": {
    "mode": "frequency_response",
    "frequency_start_hz": 1.0,
    "frequency_end_hz": 10.0,
    "frequency_steps": 5,
    "output_csv": "unused.csv"
  },
  "observations": [
    {
      "id": "obs_trunk",
      "target_type": "branch",
      "target_id": "trunk",
      "target_node": "tip",
      "target_component": "ux"
    }
  ]
}
)json";
    }

    const auto model = orchard::loadModelFromFile(json_path.string());
    check(!model.branches.empty(), "circular shorthand model should load one branch");
    const auto& profiles = model.branches.front().sectionSeries().profiles();
    check(profiles.size() == 2U, "circular shorthand branch should expose two profiles");

    const auto properties = profiles.front()->evaluate();
    const double expected_area = kPi * 0.025 * 0.025;
    checkClose(
        properties.total_area,
        expected_area,
        expected_area * 0.01,
        "circular shorthand profile area should match the reference circle"
    );

    std::filesystem::remove(json_path);
}

void testAutoNonlinearLevelInjection() {
    const std::filesystem::path json_path = std::filesystem::current_path() / "orchard_test_auto_nonlinear.json";
    {
        std::ofstream stream(json_path);
        stream << R"json(
{
  "metadata": {
    "name": "auto_nonlinear_cpp"
  },
  "materials": [
    {
      "id": "xylem_default",
      "tissue": "xylem",
      "model": "linear",
      "density": 720.0,
      "youngs_modulus": 1450000000.0,
      "poisson_ratio": 0.31,
      "damping_ratio": 0.035
    },
    {
      "id": "pith_default",
      "tissue": "pith",
      "model": "linear",
      "density": 240.0,
      "youngs_modulus": 260000000.0,
      "poisson_ratio": 0.33,
      "damping_ratio": 0.06
    },
    {
      "id": "phloem_default",
      "tissue": "phloem",
      "model": "linear",
      "density": 920.0,
      "youngs_modulus": 650000000.0,
      "poisson_ratio": 0.34,
      "damping_ratio": 0.055
    }
  ],
  "branches": [
    {
      "id": "trunk",
      "parent_branch_id": null,
      "level": 0,
      "start": [0.0, 0.0, 0.0],
      "end": [0.0, 0.0, 1.2],
      "stations": [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.05},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.04}
      ],
      "discretization": {"num_elements": 3, "hotspot": false}
    },
    {
      "id": "primary",
      "parent_branch_id": "trunk",
      "level": 1,
      "start": [0.0, 0.0, 1.0],
      "end": [0.55, 0.0, 1.45],
      "stations": [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.03},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.02}
      ],
      "discretization": {"num_elements": 2, "hotspot": false}
    },
    {
      "id": "secondary",
      "parent_branch_id": "primary",
      "level": 2,
      "start": [0.55, 0.0, 1.45],
      "end": [0.85, 0.22, 1.75],
      "stations": [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.018},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.014}
      ],
      "discretization": {"num_elements": 2, "hotspot": false}
    },
    {
      "id": "tertiary",
      "parent_branch_id": "secondary",
      "level": 3,
      "start": [0.85, 0.22, 1.75],
      "end": [1.05, 0.35, 1.98],
      "stations": [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.012},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.01}
      ],
      "discretization": {"num_elements": 2, "hotspot": false}
    }
  ],
  "fruits": [],
  "clamps": [
    {
      "branch_id": "trunk",
      "support_stiffness": 1.0,
      "support_damping": 0.0,
      "cubic_stiffness": 0.0
    }
  ],
  "excitation": {
    "kind": "harmonic_force",
    "target_branch_id": "primary",
    "target_node": "tip",
    "target_component": "ux",
    "amplitude": 1.0,
    "phase_degrees": 0.0,
    "driving_frequency_hz": 6.0
  },
  "analysis": {
    "mode": "time_history",
    "frequency_start_hz": 1.0,
    "frequency_end_hz": 12.0,
    "frequency_steps": 8,
    "time_step_seconds": 0.002,
    "total_time_seconds": 0.05,
    "output_stride": 1,
    "max_nonlinear_iterations": 12,
    "nonlinear_tolerance": 1.0e-8,
    "rayleigh_alpha": 0.0,
    "rayleigh_beta": 1.0e-4,
    "auto_nonlinear_levels": [2, 3],
    "auto_nonlinear_cubic_scale": 2200000.0,
    "output_csv": "unused.csv"
  },
  "observations": [
    {
      "id": "obs_secondary",
      "target_type": "branch",
      "target_id": "secondary",
      "target_node": "tip",
      "target_component": "ux"
    }
  ]
}
)json";
    }

    const auto model = orchard::loadModelFromFile(json_path.string());
    check(model.analysis.auto_nonlinear_levels.size() == 2U, "auto nonlinear levels should load");
    check(model.analysis.auto_nonlinear_levels[0] == 2, "first auto nonlinear level should be 2");
    checkClose(
        model.analysis.auto_nonlinear_cubic_scale,
        2200000.0,
        1.0e-9,
        "auto nonlinear cubic scale should load"
    );

    orchard::StructuralAssembler assembler;
    const auto assembled = assembler.assemble(model);

    check(assembled.system.nonlinear_links.size() == 2U, "secondary and tertiary branches should auto-inject cubic links");

    const auto has_secondary = std::find_if(
        assembled.system.nonlinear_links.begin(),
        assembled.system.nonlinear_links.end(),
        [](const orchard::NonlinearLink& link) { return link.label == "auto_joint:secondary"; }
    ) != assembled.system.nonlinear_links.end();
    const auto has_tertiary = std::find_if(
        assembled.system.nonlinear_links.begin(),
        assembled.system.nonlinear_links.end(),
        [](const orchard::NonlinearLink& link) { return link.label == "auto_joint:tertiary"; }
    ) != assembled.system.nonlinear_links.end();

    check(has_secondary, "secondary branch should receive an auto nonlinear link");
    check(has_tertiary, "tertiary branch should receive an auto nonlinear link");

    std::filesystem::remove(json_path);
}

void testCantileverBeamFirstMode() {
    const auto model = buildCantileverModel(10, 0.005);
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
    const double area = kPi * radius * radius;
    const double inertia = kPi * std::pow(radius, 4) / 4.0;
    const double omega_1 = std::pow(1.875, 2) * std::sqrt((1.0e10 * inertia) / (750.0 * area * std::pow(1.0, 4)));
    const double expected_frequency = omega_1 / (2.0 * kPi);

    checkClose(peak_frequency, expected_frequency, expected_frequency * 0.05, "cantilever first-mode peak should match Euler-Bernoulli reference");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests {
        {"section partition geometry", testSectionPartitionGeometry},
        {"material loading", testMaterialLoading},
        {"topology assembly", testTopologyAssembly},
        {"matrix assembly", testMatrixAssembly},
        {"demo response and csv output", testDemoResponseAndCsvOutput},
        {"time-history csv includes excitation channels", testTimeHistoryCsvIncludesExcitationChannels},
        {"circular shorthand loading", testCircularShorthandLoading},
        {"auto nonlinear level injection", testAutoNonlinearLevelInjection},
        {"cantilever beam first mode", testCantileverBeamFirstMode},
    };

    int passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << name << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "Passed " << passed << " tests\n";
    return 0;
}
