#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkClose(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

void testSectionPartitionGeometry() {
    using namespace orchard;

    MeasuredSectionSeries series;

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

    ParameterizedSectionProfile profile(0.0, regions);
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

    check(assembled.system.mass.rows() == model.branches.size() + model.fruits.size(), "DOF count should equal branches plus fruits");
    check(assembled.system.mass.cols() == assembled.system.mass.rows(), "mass matrix should be square");
    check(assembled.excitation_dof == assembled.branch_dofs.at("trunk"), "excitation should target trunk DOF");
    check(assembled.system.mass(assembled.branch_dofs.at("trunk"), assembled.branch_dofs.at("trunk")) > 0.0, "trunk mass must be positive");
    check(assembled.system.stiffness(assembled.branch_dofs.at("trunk"), assembled.branch_dofs.at("trunk")) > 0.0, "trunk stiffness must be positive");
    check(!assembled.system.nonlinear_links.empty(), "example model should assemble nonlinear links for joints or clamp");
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
    }

}

void testTimeHistoryAndNonlinearEffect() {
    auto model = orchard::loadModelFromFile(exampleModelPath());
    model.analysis.mode = orchard::AnalysisMode::TimeHistory;
    model.analysis.time_step_seconds = 0.002;
    model.analysis.total_time_seconds = 0.30;
    model.analysis.output_stride = 5;
    model.analysis.max_nonlinear_iterations = 25;
    model.analysis.nonlinear_tolerance = 1.0e-8;
    model.excitation.amplitude = 18.0;

    orchard::StructuralAssembler assembler;
    const auto assembled = assembler.assemble(model);

    orchard::NewmarkIntegrator integrator;
    const auto nonlinear_response = integrator.analyze(
        assembled.system,
        assembled.excitation_dof,
        model.excitation,
        model.analysis,
        assembled.observation_names,
        assembled.observation_dofs
    );

    auto linear_system = assembled.system;
    linear_system.nonlinear_links.clear();
    const auto linear_response = integrator.analyze(
        linear_system,
        assembled.excitation_dof,
        model.excitation,
        model.analysis,
        assembled.observation_names,
        assembled.observation_dofs
    );

    check(!nonlinear_response.points.empty(), "time-history response must not be empty");
    check(nonlinear_response.points.front().observation_values.size() == assembled.observation_names.size(), "time-history response should contain one value per observation");

    double max_difference = 0.0;
    for (std::size_t point_index = 0; point_index < nonlinear_response.points.size(); ++point_index) {
        const auto& nonlinear_point = nonlinear_response.points[point_index];
        const auto& linear_point = linear_response.points[point_index];
        for (std::size_t value_index = 0; value_index < nonlinear_point.observation_values.size(); ++value_index) {
            max_difference = std::max(
                max_difference,
                std::abs(nonlinear_point.observation_values[value_index] - linear_point.observation_values[value_index])
            );
        }
    }
    check(max_difference > 1.0e-7, "nonlinear links should change the transient response compared with the linearized system");

    const std::filesystem::path output_path = std::filesystem::current_path() / "orchard_test_time_history.csv";
    nonlinear_response.writeCsv(output_path.string());
    check(std::filesystem::exists(output_path), "time-history CSV should be written");

    {
        std::ifstream stream(output_path);
        std::string header;
        std::getline(stream, header);
        check(header.find("time_s") != std::string::npos, "CSV header should contain time_s");
    }
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests {
        {"section partition geometry", testSectionPartitionGeometry},
        {"material loading", testMaterialLoading},
        {"topology assembly", testTopologyAssembly},
        {"matrix assembly", testMatrixAssembly},
        {"demo response and csv output", testDemoResponseAndCsvOutput},
        {"time history and nonlinear effect", testTimeHistoryAndNonlinearEffect},
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
