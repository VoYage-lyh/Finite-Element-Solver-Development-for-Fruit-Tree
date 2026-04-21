#include "orchard_solver/io/ModelIO.h"

#include <memory>
#include <string_view>
#include <stdexcept>

namespace orchard {

namespace {

constexpr double kDefaultPithRadiusFraction = 0.15;
constexpr double kDefaultPhloemThicknessFraction = 0.08;
constexpr int kDefaultCircularSamples = 96;

const JsonValue& requireMember(const JsonValue::object_t& object, const std::string_view key) {
    const auto it = object.find(std::string(key));
    if (it == object.end()) {
        throw std::runtime_error("Missing required JSON key: " + std::string(key));
    }

    return it->second;
}

const JsonValue* optionalMember(const JsonValue::object_t& object, const std::string_view key) {
    const auto it = object.find(std::string(key));
    return it == object.end() ? nullptr : &(it->second);
}

std::string requireString(const JsonValue& value, const std::string& field_name) {
    if (!value.isString()) {
        throw std::runtime_error("JSON field must be a string: " + field_name);
    }
    return value.asString();
}

double requireNumber(const JsonValue& value, const std::string& field_name) {
    if (!value.isNumber()) {
        throw std::runtime_error("JSON field must be a number: " + field_name);
    }
    return value.asNumber();
}

int requireInt(const JsonValue& value, const std::string& field_name) {
    return static_cast<int>(requireNumber(value, field_name));
}

Vec2 parseVec2(const JsonValue& value, const std::string& field_name) {
    if (!value.isArray() || value.asArray().size() != 2U) {
        throw std::runtime_error("JSON field must be a 2D vector: " + field_name);
    }

    const auto& array = value.asArray();
    return Vec2 {
        requireNumber(array[0], field_name + "[0]"),
        requireNumber(array[1], field_name + "[1]")
    };
}

Vec3 parseVec3(const JsonValue& value, const std::string& field_name) {
    if (!value.isArray() || value.asArray().size() != 3U) {
        throw std::runtime_error("JSON field must be a 3D vector: " + field_name);
    }

    const auto& array = value.asArray();
    return Vec3 {
        requireNumber(array[0], field_name + "[0]"),
        requireNumber(array[1], field_name + "[1]"),
        requireNumber(array[2], field_name + "[2]")
    };
}

std::optional<std::string> parseOptionalString(const JsonValue::object_t& object, const std::string_view key) {
    if (const auto* value = optionalMember(object, key)) {
        if (value->isNull()) {
            return std::nullopt;
        }
        return requireString(*value, std::string(key));
    }

    return std::nullopt;
}

std::string parseNodeSelector(const JsonValue& value, const std::string& field_name) {
    if (value.isString()) {
        return value.asString();
    }
    if (value.isNumber()) {
        return std::to_string(static_cast<int>(value.asNumber()));
    }

    throw std::runtime_error("JSON field must be a node selector string or integer: " + field_name);
}

std::vector<std::string> parseObservationTargetComponents(const JsonValue::object_t& object) {
    if (const auto* target_components = optionalMember(object, "target_components")) {
        if (!target_components->isArray()) {
            throw std::runtime_error("observations[].target_components must be an array of strings");
        }

        std::vector<std::string> components;
        for (const auto& component_value : target_components->asArray()) {
            components.push_back(requireString(component_value, "observations[].target_components[]"));
        }
        if (components.empty()) {
            throw std::runtime_error("observations[].target_components must not be empty");
        }
        return components;
    }

    if (const auto* target_component = optionalMember(object, "target_component")) {
        return {requireString(*target_component, "observations[].target_component")};
    }

    return {"ux"};
}

std::vector<int> parseIntegerArray(const JsonValue& value, const std::string& field_name) {
    if (!value.isArray()) {
        throw std::runtime_error("JSON field must be an array of numbers: " + field_name);
    }

    std::vector<int> values;
    values.reserve(value.asArray().size());
    for (const auto& item : value.asArray()) {
        values.push_back(requireInt(item, field_name + "[]"));
    }
    return values;
}

ExcitationKind parseExcitationKind(const std::string& value) {
    if (value == "harmonic_force") {
        return ExcitationKind::HarmonicForce;
    }
    if (value == "harmonic_displacement") {
        return ExcitationKind::HarmonicDisplacement;
    }
    if (value == "harmonic_acceleration") {
        return ExcitationKind::HarmonicAcceleration;
    }

    throw std::runtime_error("Unsupported excitation kind: " + value);
}

AnalysisMode parseAnalysisMode(const std::string& value) {
    if (value == "frequency_response") {
        return AnalysisMode::FrequencyResponse;
    }
    if (value == "time_history") {
        return AnalysisMode::TimeHistory;
    }

    throw std::runtime_error("Unsupported analysis mode: " + value);
}

RegionGeometry parseRegionGeometry(const JsonValue& value) {
    const auto& object = value.asObject();
    RegionGeometry geometry;
    geometry.samples = optionalMember(object, "samples") ? requireInt(*optionalMember(object, "samples"), "samples") : 48;

    const std::string type = requireString(requireMember(object, "type"), "shape.type");
    if (type == "solid_ellipse") {
        geometry.kind = SectionShapeKind::SolidEllipse;
        geometry.center = parseVec2(requireMember(object, "center"), "shape.center");
        geometry.radii = parseVec2(requireMember(object, "radii"), "shape.radii");
    } else if (type == "elliptic_ring") {
        geometry.kind = SectionShapeKind::EllipticRing;
        geometry.outer_center = parseVec2(requireMember(object, "outer_center"), "shape.outer_center");
        geometry.outer_radii = parseVec2(requireMember(object, "outer_radii"), "shape.outer_radii");
        geometry.inner_center = parseVec2(requireMember(object, "inner_center"), "shape.inner_center");
        geometry.inner_radii = parseVec2(requireMember(object, "inner_radii"), "shape.inner_radii");
    } else if (type == "polygon") {
        geometry.kind = SectionShapeKind::Polygon;

        const auto parse_points = [](const JsonValue& points_value, const std::string& field_name) {
            if (!points_value.isArray()) {
                throw std::runtime_error("JSON field must be an array of points: " + field_name);
            }

            std::vector<Vec2> points;
            for (const auto& point_value : points_value.asArray()) {
                points.push_back(parseVec2(point_value, field_name));
            }
            return points;
        };

        geometry.outer_points = parse_points(requireMember(object, "outer_points"), "shape.outer_points");
        if (const auto* inner_points = optionalMember(object, "inner_points")) {
            geometry.inner_points = parse_points(*inner_points, "shape.inner_points");
        }
    } else {
        throw std::runtime_error("Unsupported region shape type: " + type);
    }

    return geometry;
}

std::vector<TissueRegionDefinition> parseRegions(const JsonValue& value) {
    if (!value.isArray()) {
        throw std::runtime_error("Station regions must be an array");
    }

    std::vector<TissueRegionDefinition> regions;
    for (const auto& region_value : value.asArray()) {
        const auto& object = region_value.asObject();
        TissueRegionDefinition region;
        region.tissue = tissueTypeFromString(requireString(requireMember(object, "tissue"), "region.tissue"));
        region.material_id = requireString(requireMember(object, "material_id"), "region.material_id");
        region.geometry = parseRegionGeometry(requireMember(object, "shape"));
        regions.push_back(std::move(region));
    }

    return regions;
}

std::vector<TissueRegionDefinition> buildDefaultCircularRegions(const double outer_radius) {
    if (outer_radius <= 0.0) {
        throw std::runtime_error("station.outer_radius must be positive for circular shorthand");
    }

    const double pith_radius = outer_radius * kDefaultPithRadiusFraction;
    const double xylem_outer_radius = outer_radius * (1.0 - kDefaultPhloemThicknessFraction);
    if (xylem_outer_radius <= pith_radius) {
        throw std::runtime_error("Circular shorthand leaves no room for the xylem ring");
    }

    RegionGeometry pith_geometry;
    pith_geometry.kind = SectionShapeKind::SolidEllipse;
    pith_geometry.center = Vec2 {0.0, 0.0};
    pith_geometry.radii = Vec2 {pith_radius, pith_radius};
    pith_geometry.samples = kDefaultCircularSamples;

    RegionGeometry xylem_geometry;
    xylem_geometry.kind = SectionShapeKind::EllipticRing;
    xylem_geometry.outer_center = Vec2 {0.0, 0.0};
    xylem_geometry.outer_radii = Vec2 {xylem_outer_radius, xylem_outer_radius};
    xylem_geometry.inner_center = Vec2 {0.0, 0.0};
    xylem_geometry.inner_radii = Vec2 {pith_radius, pith_radius};
    xylem_geometry.samples = kDefaultCircularSamples;

    RegionGeometry phloem_geometry;
    phloem_geometry.kind = SectionShapeKind::EllipticRing;
    phloem_geometry.outer_center = Vec2 {0.0, 0.0};
    phloem_geometry.outer_radii = Vec2 {outer_radius, outer_radius};
    phloem_geometry.inner_center = Vec2 {0.0, 0.0};
    phloem_geometry.inner_radii = Vec2 {xylem_outer_radius, xylem_outer_radius};
    phloem_geometry.samples = kDefaultCircularSamples;

    return {
        TissueRegionDefinition {TissueType::Pith, "pith_default", pith_geometry},
        TissueRegionDefinition {TissueType::Xylem, "xylem_default", xylem_geometry},
        TissueRegionDefinition {TissueType::Phloem, "phloem_default", phloem_geometry},
    };
}

void requireCircularShorthandMaterials(const MaterialLibrary& materials) {
    for (const auto& material_id : {"xylem_default", "pith_default", "phloem_default"}) {
        if (!materials.contains(material_id)) {
            throw std::runtime_error(
                "Circular shorthand requires material to be defined: " + std::string(material_id)
            );
        }
    }
}

std::shared_ptr<CrossSectionProfile> parseStationProfile(
    const JsonValue& value,
    const MaterialLibrary& materials
) {
    const auto& object = value.asObject();
    const double station = requireNumber(requireMember(object, "s"), "station.s");

    if (const auto* shorthand_value = optionalMember(object, "shorthand")) {
        const std::string shorthand = requireString(*shorthand_value, "station.shorthand");
        if (shorthand != "circular") {
            throw std::runtime_error("Unsupported station shorthand: " + shorthand);
        }
        requireCircularShorthandMaterials(materials);
        const auto regions = buildDefaultCircularRegions(
            requireNumber(requireMember(object, "outer_radius"), "station.outer_radius")
        );
        return std::make_shared<ParameterizedSectionProfile>(station, regions);
    }

    const std::string profile_type = optionalMember(object, "profile_type")
        ? requireString(*optionalMember(object, "profile_type"), "station.profile_type")
        : "parameterized";

    const auto regions = parseRegions(requireMember(object, "regions"));

    if (profile_type == "parameterized") {
        return std::make_shared<ParameterizedSectionProfile>(station, regions);
    }
    if (profile_type == "contour") {
        return std::make_shared<ContourSectionProfile>(station, regions);
    }

    throw std::runtime_error("Unsupported profile_type: " + profile_type);
}

} // namespace

OrchardModel TreeModelBuilder::build(const JsonValue& root) const {
    if (!root.isObject()) {
        throw std::runtime_error("Root JSON value must be an object");
    }

    const auto& object = root.asObject();
    OrchardModel model;

    if (const auto* metadata_value = optionalMember(object, "metadata")) {
        const auto& metadata = metadata_value->asObject();
        if (const auto* name = optionalMember(metadata, "name")) {
            model.metadata.name = requireString(*name, "metadata.name");
        }
        if (const auto* cultivar = optionalMember(metadata, "cultivar")) {
            model.metadata.cultivar = requireString(*cultivar, "metadata.cultivar");
        }
    }

    const auto& materials_value = requireMember(object, "materials");
    const auto& materials_array = materials_value.asArray();
    for (const auto& material_value : materials_array) {
        const auto& material_object = material_value.asObject();
        MaterialProperties properties;
        properties.id = requireString(requireMember(material_object, "id"), "materials[].id");
        properties.tissue = tissueTypeFromString(requireString(requireMember(material_object, "tissue"), "materials[].tissue"));
        properties.density = requireNumber(requireMember(material_object, "density"), "materials[].density");
        properties.youngs_modulus = requireNumber(requireMember(material_object, "youngs_modulus"), "materials[].youngs_modulus");
        if (const auto* poisson_ratio = optionalMember(material_object, "poisson_ratio")) {
            properties.poisson_ratio = requireNumber(*poisson_ratio, "materials[].poisson_ratio");
        }
        properties.damping_ratio = requireNumber(requireMember(material_object, "damping_ratio"), "materials[].damping_ratio");
        if (const auto* nonlinear_alpha = optionalMember(material_object, "nonlinear_alpha")) {
            properties.nonlinear_alpha = requireNumber(*nonlinear_alpha, "materials[].nonlinear_alpha");
        }

        const std::string model_type = optionalMember(material_object, "model")
            ? requireString(*optionalMember(material_object, "model"), "materials[].model")
            : "linear";

        if (model_type == "linear") {
            model.materials.addLinearElastic(properties);
        } else if (model_type == "nonlinear") {
            model.materials.addNonlinearElastic(properties);
        } else if (model_type == "orthotropic_placeholder") {
            properties.orthotropic_enabled = true;
            model.materials.addOrthotropicPlaceholder(properties);
        } else {
            throw std::runtime_error("Unsupported material model: " + model_type);
        }
    }

    const auto& branches_value = requireMember(object, "branches");
    const auto& branches_array = branches_value.asArray();
    for (const auto& branch_value : branches_array) {
        const auto& branch_object = branch_value.asObject();
        const std::string branch_id = requireString(requireMember(branch_object, "id"), "branches[].id");
        const auto parent_branch_id = parseOptionalString(branch_object, "parent_branch_id");
        const int level = requireInt(requireMember(branch_object, "level"), "branches[].level");
        const BranchPath path(
            parseVec3(requireMember(branch_object, "start"), "branches[].start"),
            parseVec3(requireMember(branch_object, "end"), "branches[].end")
        );

        MeasuredSectionSeries series;
        const auto& stations_value = requireMember(branch_object, "stations");
        const auto& stations_array = stations_value.asArray();
        for (const auto& station_value : stations_array) {
            series.addProfile(parseStationProfile(station_value, model.materials));
        }

        BranchDiscretizationHint hint;
        if (const auto* discretization = optionalMember(branch_object, "discretization")) {
            const auto& discretization_object = discretization->asObject();
            if (const auto* num_elements = optionalMember(discretization_object, "num_elements")) {
                hint.num_elements = requireInt(*num_elements, "branches[].discretization.num_elements");
            }
            if (const auto* hotspot = optionalMember(discretization_object, "hotspot")) {
                if (!hotspot->isBool()) {
                    throw std::runtime_error("branches[].discretization.hotspot must be boolean");
                }
                hint.hotspot = hotspot->asBool();
            }
        }

        model.topology.addBranch(branch_id, parent_branch_id, level, path);
        model.branches.emplace_back(branch_id, parent_branch_id, level, path, std::move(series), hint);
    }
    model.topology.rebuildChildLinks();

    if (const auto* joints_value = optionalMember(object, "joints")) {
        for (const auto& joint_value : joints_value->asArray()) {
            const auto& joint_object = joint_value.asObject();
            JointComponent joint;
            joint.id = requireString(requireMember(joint_object, "id"), "joints[].id");
            joint.parent_branch_id = requireString(requireMember(joint_object, "parent_branch_id"), "joints[].parent_branch_id");
            joint.child_branch_id = requireString(requireMember(joint_object, "child_branch_id"), "joints[].child_branch_id");
            if (const auto* linear_scale = optionalMember(joint_object, "linear_stiffness_scale")) {
                joint.linear_stiffness_scale = requireNumber(*linear_scale, "joints[].linear_stiffness_scale");
            }

            if (const auto* law_value = optionalMember(joint_object, "law")) {
                const auto& law_object = law_value->asObject();
                const std::string law_type = requireString(requireMember(law_object, "type"), "joints[].law.type");
                if (law_type == "polynomial") {
                    joint.law = std::make_shared<PolynomialRotationalJointLaw>(
                        requireNumber(requireMember(law_object, "linear_scale"), "joints[].law.linear_scale"),
                        requireNumber(requireMember(law_object, "cubic_scale"), "joints[].law.cubic_scale")
                    );
                } else if (law_type == "gap_friction") {
                    joint.law = std::make_shared<GapFrictionJointLaw>(
                        requireNumber(requireMember(law_object, "closed_scale"), "joints[].law.closed_scale"),
                        requireNumber(requireMember(law_object, "open_scale"), "joints[].law.open_scale"),
                        requireNumber(requireMember(law_object, "gap_threshold"), "joints[].law.gap_threshold")
                    );
                } else {
                    throw std::runtime_error("Unsupported joint law type: " + law_type);
                }
            }

            model.joints.push_back(std::move(joint));
        }
    }

    if (const auto* fruits_value = optionalMember(object, "fruits")) {
        for (const auto& fruit_value : fruits_value->asArray()) {
            const auto& fruit_object = fruit_value.asObject();
            FruitAttachment fruit;
            fruit.id = requireString(requireMember(fruit_object, "id"), "fruits[].id");
            fruit.branch_id = requireString(requireMember(fruit_object, "branch_id"), "fruits[].branch_id");
            fruit.location_s = requireNumber(requireMember(fruit_object, "location_s"), "fruits[].location_s");
            fruit.mass = requireNumber(requireMember(fruit_object, "mass"), "fruits[].mass");
            fruit.stiffness = requireNumber(requireMember(fruit_object, "stiffness"), "fruits[].stiffness");
            fruit.damping = requireNumber(requireMember(fruit_object, "damping"), "fruits[].damping");
            model.fruits.push_back(std::move(fruit));
        }
    }

    if (const auto* clamps_value = optionalMember(object, "clamps")) {
        for (const auto& clamp_value : clamps_value->asArray()) {
            const auto& clamp_object = clamp_value.asObject();
            ClampBoundaryCondition clamp;
            clamp.branch_id = requireString(requireMember(clamp_object, "branch_id"), "clamps[].branch_id");
            clamp.support_stiffness = requireNumber(requireMember(clamp_object, "support_stiffness"), "clamps[].support_stiffness");
            clamp.support_damping = requireNumber(requireMember(clamp_object, "support_damping"), "clamps[].support_damping");
            if (const auto* cubic_stiffness = optionalMember(clamp_object, "cubic_stiffness")) {
                clamp.cubic_stiffness = requireNumber(*cubic_stiffness, "clamps[].cubic_stiffness");
            }
            model.clamps.push_back(std::move(clamp));
        }
    }

    {
        const auto& excitation_value = requireMember(object, "excitation");
        const auto& excitation_object = excitation_value.asObject();
        model.excitation.kind = parseExcitationKind(requireString(requireMember(excitation_object, "kind"), "excitation.kind"));
        model.excitation.target_branch_id = requireString(requireMember(excitation_object, "target_branch_id"), "excitation.target_branch_id");
        if (const auto* target_node = optionalMember(excitation_object, "target_node")) {
            model.excitation.target_node = parseNodeSelector(*target_node, "excitation.target_node");
        }
        if (const auto* target_component = optionalMember(excitation_object, "target_component")) {
            model.excitation.target_component = requireString(*target_component, "excitation.target_component");
        }
        model.excitation.amplitude = requireNumber(requireMember(excitation_object, "amplitude"), "excitation.amplitude");
        model.excitation.phase_degrees = requireNumber(requireMember(excitation_object, "phase_degrees"), "excitation.phase_degrees");
        if (const auto* driving_frequency_hz = optionalMember(excitation_object, "driving_frequency_hz")) {
            model.excitation.driving_frequency_hz = requireNumber(*driving_frequency_hz, "excitation.driving_frequency_hz");
        }
    }

    {
        const auto& analysis_value = requireMember(object, "analysis");
        const auto& analysis_object = analysis_value.asObject();
        if (const auto* mode = optionalMember(analysis_object, "mode")) {
            model.analysis.mode = parseAnalysisMode(requireString(*mode, "analysis.mode"));
        }
        model.analysis.frequency_start_hz = requireNumber(requireMember(analysis_object, "frequency_start_hz"), "analysis.frequency_start_hz");
        model.analysis.frequency_end_hz = requireNumber(requireMember(analysis_object, "frequency_end_hz"), "analysis.frequency_end_hz");
        model.analysis.frequency_steps = requireInt(requireMember(analysis_object, "frequency_steps"), "analysis.frequency_steps");
        if (const auto* time_step_seconds = optionalMember(analysis_object, "time_step_seconds")) {
            model.analysis.time_step_seconds = requireNumber(*time_step_seconds, "analysis.time_step_seconds");
        }
        if (const auto* total_time_seconds = optionalMember(analysis_object, "total_time_seconds")) {
            model.analysis.total_time_seconds = requireNumber(*total_time_seconds, "analysis.total_time_seconds");
        }
        if (const auto* output_stride = optionalMember(analysis_object, "output_stride")) {
            model.analysis.output_stride = requireInt(*output_stride, "analysis.output_stride");
        }
        if (const auto* max_nonlinear_iterations = optionalMember(analysis_object, "max_nonlinear_iterations")) {
            model.analysis.max_nonlinear_iterations = requireInt(*max_nonlinear_iterations, "analysis.max_nonlinear_iterations");
        }
        if (const auto* nonlinear_tolerance = optionalMember(analysis_object, "nonlinear_tolerance")) {
            model.analysis.nonlinear_tolerance = requireNumber(*nonlinear_tolerance, "analysis.nonlinear_tolerance");
        }
        if (const auto* rayleigh_alpha = optionalMember(analysis_object, "rayleigh_alpha")) {
            model.analysis.rayleigh_alpha = requireNumber(*rayleigh_alpha, "analysis.rayleigh_alpha");
        }
        if (const auto* rayleigh_beta = optionalMember(analysis_object, "rayleigh_beta")) {
            model.analysis.rayleigh_beta = requireNumber(*rayleigh_beta, "analysis.rayleigh_beta");
        }
        if (const auto* auto_nonlinear_levels = optionalMember(analysis_object, "auto_nonlinear_levels")) {
            model.analysis.auto_nonlinear_levels = parseIntegerArray(
                *auto_nonlinear_levels,
                "analysis.auto_nonlinear_levels"
            );
        }
        if (const auto* auto_nonlinear_cubic_scale = optionalMember(analysis_object, "auto_nonlinear_cubic_scale")) {
            model.analysis.auto_nonlinear_cubic_scale = requireNumber(
                *auto_nonlinear_cubic_scale,
                "analysis.auto_nonlinear_cubic_scale"
            );
        }
        if (const auto* include_gravity_prestress = optionalMember(analysis_object, "include_gravity_prestress")) {
            if (!include_gravity_prestress->isBool()) {
                throw std::runtime_error("analysis.include_gravity_prestress must be boolean");
            }
            model.analysis.include_gravity_prestress = include_gravity_prestress->asBool();
        }
        if (const auto* gravity_direction = optionalMember(analysis_object, "gravity_direction")) {
            model.analysis.gravity_direction = parseVec3(*gravity_direction, "analysis.gravity_direction");
        }
        if (const auto* output_csv = optionalMember(analysis_object, "output_csv")) {
            model.analysis.output_csv = requireString(*output_csv, "analysis.output_csv");
        }
    }

    if (const auto* observations_value = optionalMember(object, "observations")) {
        for (const auto& observation_value : observations_value->asArray()) {
            const auto& observation_object = observation_value.asObject();
            ObservationPoint observation;
            observation.id = requireString(requireMember(observation_object, "id"), "observations[].id");
            observation.target_type = requireString(requireMember(observation_object, "target_type"), "observations[].target_type");
            observation.target_id = requireString(requireMember(observation_object, "target_id"), "observations[].target_id");
            if (const auto* target_node = optionalMember(observation_object, "target_node")) {
                observation.target_node = parseNodeSelector(*target_node, "observations[].target_node");
            }
            observation.target_components = parseObservationTargetComponents(observation_object);
            model.observations.push_back(std::move(observation));
        }
    }

    return model;
}

void ModelValidator::validate(const OrchardModel& model) const {
    if (model.materials.ids().empty()) {
        throw std::runtime_error("Model must define at least one material");
    }

    std::string topology_error;
    if (!model.topology.validate(topology_error)) {
        throw std::runtime_error("Invalid topology: " + topology_error);
    }

    if (model.branches.empty()) {
        throw std::runtime_error("Model must define at least one branch");
    }

    for (const auto& branch : model.branches) {
        if (branch.sectionSeries().profiles().empty()) {
            throw std::runtime_error("Branch '" + branch.id() + "' has no section stations");
        }
        if (branch.discretizationHint().num_elements < 1) {
            throw std::runtime_error("Branch '" + branch.id() + "' must use at least one beam element");
        }

        for (const auto& profile : branch.sectionSeries().profiles()) {
            const auto geometry = profile->evaluate();
            for (const auto& region : geometry.regions) {
                if (!model.materials.contains(region.material_id)) {
                    throw std::runtime_error("Branch '" + branch.id() + "' references unknown material '" + region.material_id + "'");
                }
            }
        }
    }

    if (!model.topology.contains(model.excitation.target_branch_id)) {
        throw std::runtime_error("Excitation target branch not found: " + model.excitation.target_branch_id);
    }
    if (
        model.excitation.target_component != "ux"
        && model.excitation.target_component != "uy"
        && model.excitation.target_component != "uz"
    ) {
        throw std::runtime_error("excitation.target_component must be one of ux, uy, uz");
    }
    if (model.analysis.frequency_steps < 1) {
        throw std::runtime_error("analysis.frequency_steps must be at least 1");
    }
    if (model.analysis.time_step_seconds <= 0.0) {
        throw std::runtime_error("analysis.time_step_seconds must be positive");
    }
    if (model.analysis.total_time_seconds <= 0.0) {
        throw std::runtime_error("analysis.total_time_seconds must be positive");
    }
    if (model.analysis.output_stride < 1) {
        throw std::runtime_error("analysis.output_stride must be at least 1");
    }
    if (model.analysis.max_nonlinear_iterations < 1) {
        throw std::runtime_error("analysis.max_nonlinear_iterations must be at least 1");
    }
    if (model.analysis.nonlinear_tolerance <= 0.0) {
        throw std::runtime_error("analysis.nonlinear_tolerance must be positive");
    }
    if (model.analysis.rayleigh_alpha < 0.0 || model.analysis.rayleigh_beta < 0.0) {
        throw std::runtime_error("analysis.rayleigh_alpha and analysis.rayleigh_beta must be non-negative");
    }

    for (const auto& clamp : model.clamps) {
        if (!model.topology.contains(clamp.branch_id)) {
            throw std::runtime_error("Clamp references unknown branch: " + clamp.branch_id);
        }
    }

    for (const auto& fruit : model.fruits) {
        if (!model.topology.contains(fruit.branch_id)) {
            throw std::runtime_error("Fruit references unknown branch: " + fruit.branch_id);
        }
        if (fruit.mass <= 0.0) {
            throw std::runtime_error("Fruit mass must be positive for fruit: " + fruit.id);
        }
    }

    for (const auto& observation : model.observations) {
        if (observation.target_type == "branch") {
            if (!model.topology.contains(observation.target_id)) {
                throw std::runtime_error("Observation references unknown branch: " + observation.target_id);
            }
            for (const auto& component : observation.target_components) {
                if (
                    component != "ux"
                    && component != "uy"
                    && component != "uz"
                ) {
                    throw std::runtime_error("Branch observation target_components must contain only ux, uy, uz");
                }
            }
        } else if (observation.target_type == "fruit") {
            bool found = false;
            for (const auto& fruit : model.fruits) {
                if (fruit.id == observation.target_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("Observation references unknown fruit: " + observation.target_id);
            }
        } else {
            throw std::runtime_error("Unsupported observation target type: " + observation.target_type);
        }
    }
}

OrchardModel loadModelFromFile(const std::string& file_path) {
    const JsonValue root = parseJsonFile(file_path);
    TreeModelBuilder builder;
    ModelValidator validator;
    OrchardModel model = builder.build(root);
    validator.validate(model);
    return model;
}

} // namespace orchard
